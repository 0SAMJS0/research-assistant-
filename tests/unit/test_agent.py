"""LangGraph orchestration mechanics with a scripted model and stub tools (MCP transport is covered in
tests/integration)."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException

from app.agent.graph import AgentUnavailableError, ResearchAgent, repair_history
from tests.helpers import ScriptedChatModel, make_settings, tool_call


class StubProvider:
    def __init__(self, tools):
        self.tools = tools
        self.calls = 0

    async def get_tools(self, refresh=False):
        self.calls += 1
        return self.tools


def stub_tools(log):
    async def search_papers(query: str) -> tuple[str, dict]:
        log.append(("search_papers", query))
        return '{"papers": 1}', {
            "structured_content": {
                "papers": [
                    {
                        "arxiv_id": "1609.02907v4",
                        "title": "GCN",
                        "authors": ["Kipf"],
                        "url": "https://arxiv.org/abs/1609.02907",
                        "published": "2016-09-09",
                    }
                ]
            }
        }

    async def fetch_experiments(metrics: list[str] | None = None) -> tuple[str, dict]:
        log.append(("fetch_experiments", metrics))
        return '{"comparisons": []}', {"structured_content": {"files": ["a.csv"], "comparisons": []}}

    async def failing_tool() -> str:
        raise ToolException("upstream API exploded")

    async def slow_tool() -> str:
        await asyncio.sleep(5)
        return "late"

    def make(fn, fmt="content_and_artifact"):
        return StructuredTool.from_function(
            coroutine=fn,
            name=fn.__name__,
            description=fn.__name__,
            response_format=fmt,
            handle_tool_error=lambda e: f"Error: {e}",
        )

    return [
        make(search_papers),
        make(fetch_experiments),
        make(failing_tool, "content"),
        make(slow_tool, "content"),
    ]


def agent(tmp_path, steps, log, **overrides):
    settings = make_settings(tmp_path, **overrides)
    model = ScriptedChatModel(steps=steps)
    return ResearchAgent(model, StubProvider(stub_tools(log)), settings), model


async def test_multi_tool_orchestration_and_trace(tmp_path):
    log = []
    a, model = agent(
        tmp_path,
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("search_papers", {"query": "gnn"}, "c1"),
                    tool_call("fetch_experiments", {"metrics": ["val_accuracy"]}, "c2"),
                ],
            ),
            lambda msgs: AIMessage(
                content=f"Saw {sum(isinstance(m, ToolMessage) for m in msgs)} results. "
                "GCN https://arxiv.org/abs/1609.02907"
            ),
        ],
        log,
    )
    result = await a.run("compare", "thread-0001")
    assert set(model.bound_tools) == {"search_papers", "fetch_experiments", "failing_tool", "slow_tool"}
    assert sorted(log) == [("fetch_experiments", ["val_accuracy"]), ("search_papers", "gnn")]
    assert [t["tool"] for t in result.trace] == ["search_papers", "fetch_experiments"]
    assert all(t["status"] == "success" and t["duration_ms"] >= 0 for t in result.trace)
    assert result.answer.startswith("Saw 2 results")
    assert result.tool_results["search_papers"][0]["papers"][0]["title"] == "GCN"
    assert result.citation_check == {"verified": ["arXiv:1609.02907"], "unverified": []}


async def test_sequential_tool_use_feeds_results_back(tmp_path):
    log = []
    a, model = agent(
        tmp_path,
        [
            AIMessage(content="", tool_calls=[tool_call("search_papers", {"query": "gat"}, "a")]),
            AIMessage(content="", tool_calls=[tool_call("fetch_experiments", {}, "b")]),
            AIMessage(content="done"),
        ],
        log,
    )
    result = await a.run("go")
    assert [t["tool"] for t in result.trace] == ["search_papers", "fetch_experiments"]
    third_call = model.calls[2]
    assert [m.tool_call_id for m in third_call if isinstance(m, ToolMessage)] == ["a", "b"]


async def test_failed_and_unknown_tools_do_not_break_conversation(tmp_path):
    a, _ = agent(
        tmp_path,
        [
            AIMessage(
                content="", tool_calls=[tool_call("failing_tool", {}, "f"), tool_call("nope", {}, "n")]
            ),
            lambda msgs: AIMessage(
                content="Recovered: " + " | ".join(m.text for m in msgs if isinstance(m, ToolMessage))
            ),
        ],
        [],
    )
    result = await a.run("try")
    statuses = {t["tool"]: t["status"] for t in result.trace}
    assert statuses == {"failing_tool": "error", "nope": "error"}
    assert "upstream API exploded" in result.answer and "Unknown tool" in result.answer


async def test_tool_timeout(tmp_path):
    a, _ = agent(
        tmp_path,
        [AIMessage(content="", tool_calls=[tool_call("slow_tool", {}, "s")]), AIMessage(content="ok")],
        [],
        agent_tool_timeout_s=1,
    )
    result = await a.run("slow")
    assert result.trace[0]["status"] == "timeout" and result.answer == "ok"


async def test_tool_budget_is_enforced(tmp_path):
    log = []
    calls = [tool_call("search_papers", {"query": f"q{i}"}, f"id{i}") for i in range(3)]
    a, model = agent(
        tmp_path,
        [
            AIMessage(content="", tool_calls=calls),
            AIMessage(content="", tool_calls=[tool_call("search_papers", {"query": "again"}, "x")]),
            AIMessage(content="final"),
        ],
        log,
        agent_max_tool_calls=2,
    )
    result = await a.run("many")
    assert [t["status"] for t in result.trace] == ["success", "success", "skipped"]
    assert len(log) == 2  # the third call never executed
    # After the budget is spent the model is called without tools; a tool request it still emits is dropped.
    assert "exhausted" in model.calls[-1][0].content
    assert result.answer == "Tool budget exhausted." and ("search_papers", "again") not in log
    assert len(result.trace) == 3


async def test_recursion_limit(tmp_path):
    loop = [
        AIMessage(content="", tool_calls=[tool_call("fetch_experiments", {}, f"r{i}")]) for i in range(30)
    ]
    a, _ = agent(tmp_path, loop, [], agent_recursion_limit=5, agent_max_tool_calls=50)
    result = await a.run("loop")
    assert "step limit" in result.answer and result.stopped_reason


async def test_request_timeout(tmp_path):
    async def slow(msgs):
        raise AssertionError("unreachable")

    a, model = agent(
        tmp_path,
        [AIMessage(content="", tool_calls=[tool_call("slow_tool", {}, "s")])],
        [],
        agent_request_timeout_s=5,
        agent_tool_timeout_s=30,
    )
    a._settings.agent_request_timeout_s = 0.5
    result = await a.run("hang", "thread-timeout")
    assert "timed out" in result.answer
    # The next turn on the same thread still works (dangling tool call is repaired).
    model.steps.append(AIMessage(content="recovered"))
    second = await a.run("again", "thread-timeout")
    assert second.answer == "recovered"


async def test_multi_turn_memory(tmp_path):
    a, model = agent(tmp_path, [AIMessage(content="first"), AIMessage(content="second")], [])
    await a.run("hello", "thread-mem1")
    await a.run("follow up", "thread-mem1")
    humans = [m.content for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert humans == ["hello", "follow up"]
    await a.run("other", "thread-mem2") if model.steps.append(AIMessage(content="x")) is None else None
    assert [m.content for m in model.calls[2] if isinstance(m, HumanMessage)] == ["other"]


async def test_no_tool_answer_has_empty_trace(tmp_path):
    a, _ = agent(tmp_path, [AIMessage(content="Invented [fake.pdf, p. 2]")], [])
    result = await a.run("hi")
    assert result.trace == [] and result.citation_check["unverified"] == ["[fake.pdf, p. 2]"]
    assert "Citation check" in result.answer


async def test_unavailable_without_model(tmp_path):
    a = ResearchAgent(None, StubProvider([]), make_settings(tmp_path))
    with pytest.raises(AgentUnavailableError):
        await a.run("x")


def test_repair_history_closes_dangling_calls_and_windows():
    msgs = [
        HumanMessage("a", id="1"),
        AIMessage("", tool_calls=[tool_call("t", {}, "z")], id="2"),
        HumanMessage("b", id="3"),
    ]
    repaired = repair_history(msgs, 40)
    assert isinstance(repaired[2], ToolMessage) and repaired[2].tool_call_id == "z"
    long = [HumanMessage(str(i)) if i % 2 == 0 else AIMessage(str(i)) for i in range(50)]
    windowed = repair_history(long, 10)
    assert isinstance(windowed[0], HumanMessage) and len(windowed) <= 10
