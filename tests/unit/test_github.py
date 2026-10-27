import httpx
import pytest

from app.services.github import GitHubClient, detect_frameworks, parse_dependencies, parse_repository
from app.services.http import ExternalServiceError, build_client
from tests.helpers import github_handler


@pytest.mark.parametrize(
    "value",
    [
        "octo/tiny-gnn",
        "https://github.com/octo/tiny-gnn",
        "https://github.com/octo/tiny-gnn.git",
        "github.com/octo/tiny-gnn/tree/main/src",
        "http://www.github.com/octo/tiny-gnn/",
    ],
)
def test_parse_repository_variants(value):
    assert parse_repository(value) == ("octo", "tiny-gnn")


@pytest.mark.parametrize("value", ["", "just-a-name", "https://gitlab.com/a/b", "../../etc", "a/b c"])
def test_parse_repository_rejects_invalid(value):
    with pytest.raises(ValueError):
        parse_repository(value)


def test_dependency_parsing_and_framework_detection():
    deps = parse_dependencies("requirements.txt", "torch>=2.0\n# comment\n-e .\ntorch-geometric==2.5\n")
    assert deps == ["torch", "torch-geometric"]
    pyproject = '[project]\ndependencies = ["jax>=0.4", "flax", "optax==0.2"]\n'
    assert {"jax", "flax", "optax"} <= set(parse_dependencies("pyproject.toml", pyproject))
    assert detect_frameworks(["import tensorflow as tf", "from sklearn.svm import SVC"]) == [
        "TensorFlow",
        "scikit-learn",
    ]


async def test_analyze_repository(settings):
    calls: list[httpx.Request] = []
    gh = GitHubClient(settings, transport=httpx.MockTransport(github_handler(calls)))
    result = await gh.analyze("https://github.com/octo/tiny-gnn")
    assert result.repository == "octo/tiny-gnn" and result.stars == 42 and result.license == "MIT"
    assert result.languages == {"Python": 90.0, "Shell": 10.0}
    assert "torch" in result.dependencies and "torch-geometric" in result.dependencies
    assert {"PyTorch", "PyTorch Geometric"} <= set(result.ml_frameworks)
    assert "Graph convolution (GCN)" in result.key_algorithms
    gcn = next(r for r in result.code_references if r.symbol == "GCN")
    assert gcn.kind == "model class" and gcn.line == 5
    assert gcn.url == f"https://github.com/octo/tiny-gnn/blob/{'a' * 40}/src/model.py#L5"
    assert any("GitHub Actions" in line for line in result.architecture_summary)
    assert any("Test files detected: 1" in line for line in result.architecture_summary)
    # Secret-looking files are never requested and the client is unauthenticated by default.
    assert not any(".env" in str(c.url) for c in calls)
    assert all("authorization" not in c.headers for c in calls)
    assert all(c.url.host == "api.github.com" for c in calls)


async def test_token_is_sent_when_configured(tmp_path):
    from tests.helpers import make_settings

    calls: list[httpx.Request] = []
    gh = GitHubClient(
        make_settings(tmp_path, github_token="ghp_" + "x" * 36),
        transport=httpx.MockTransport(github_handler(calls)),
    )
    await gh.analyze("octo/tiny-gnn")
    assert calls[0].headers["authorization"].startswith("Bearer ghp_")


async def test_repository_not_found(settings):
    gh = GitHubClient(settings, transport=httpx.MockTransport(github_handler()))
    with pytest.raises(ExternalServiceError, match="not found"):
        await gh.analyze("octo/missing")


async def test_rate_limit_message(settings):
    gh = GitHubClient(settings, transport=httpx.MockTransport(github_handler(rate_limited=True)))
    with pytest.raises(ExternalServiceError, match=r"rate limit exhausted.*GITHUB_TOKEN"):
        await gh.analyze("octo/tiny-gnn")


async def test_network_timeout(settings):
    def handler(request):
        raise httpx.ConnectTimeout("down", request=request)

    gh = GitHubClient(settings, transport=httpx.MockTransport(handler))
    with pytest.raises(ExternalServiceError, match="timed out"):
        await gh.analyze("octo/tiny-gnn")


def test_outbound_client_refuses_non_allowlisted_hosts():
    with pytest.raises(ValueError):
        build_client("http://api.github.com", {"api.github.com"}, 5)
    with pytest.raises(ValueError):
        build_client("https://169.254.169.254", {"api.github.com"}, 5)


async def test_redirect_to_other_host_is_blocked():
    def handler(request):
        if request.url.host == "api.github.com":
            return httpx.Response(302, headers={"Location": "https://169.254.169.254/latest/meta-data"})
        return httpx.Response(200, text="secret")

    async with build_client(
        "https://api.github.com", {"api.github.com"}, 5, transport=httpx.MockTransport(handler)
    ) as c:
        with pytest.raises(ExternalServiceError, match="Blocked"):
            await c.get("/repos/a/b")
