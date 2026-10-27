from datetime import date

import httpx
import pytest

from app.services.arxiv import ArxivClient, build_search_query, normalize_arxiv_id, parse_feed
from app.services.http import ExternalServiceError
from tests.helpers import arxiv_handler, atom_entry, atom_feed


def client(settings, handler) -> ArxivClient:
    return ArxivClient(settings, transport=httpx.MockTransport(handler))


def test_build_search_query_plain_terms_and_dates():
    q = build_search_query("graph neural networks", date(2023, 1, 1), date(2023, 12, 31))
    assert q == (
        "(all:graph AND all:neural AND all:networks) AND submittedDate:[202301010000 TO 202312312359]"
    )


def test_build_search_query_keeps_field_prefixes_and_strips_quotes():
    assert build_search_query('ti:"attention" AND au:veličković') == "ti: attention AND au:veličković"
    with pytest.raises(ValueError):
        build_search_query("   ")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1609.02907", "1609.02907"),
        ("https://arxiv.org/abs/1609.02907v4", "1609.02907v4"),
        ("arXiv:1710.10903", "1710.10903"),
        ("https://arxiv.org/pdf/1706.02216.pdf", "1706.02216"),
        ("cs/0112017", "cs/0112017"),
    ],
)
def test_normalize_arxiv_id(raw, expected):
    assert normalize_arxiv_id(raw) == expected


@pytest.mark.parametrize("bad", ["../etc/passwd", "http://evil.com/abs/1", "1234"])
def test_normalize_arxiv_id_rejects_invalid(bad):
    with pytest.raises(ValueError):
        normalize_arxiv_id(bad)


def test_parse_feed_extracts_metadata():
    total, papers = parse_feed(atom_feed([atom_entry("1609.02907", "GCN paper")], total=7))
    assert total == 7
    p = papers[0]
    assert (p.arxiv_id, p.title, p.published) == ("1609.02907v1", "GCN paper", "2023-05-01")
    assert p.url == "https://arxiv.org/abs/1609.02907"
    assert p.authors == ["Ada Lovelace", "Alan Turing"] and p.categories == ["cs.LG"]


def test_parse_feed_malformed_and_api_error():
    with pytest.raises(ExternalServiceError, match="malformed"):
        parse_feed("<feed><not-closed>")
    error_entry = (
        "<entry><id>http://arxiv.org/api/errors#incorrect_id</id><title>Error</title>"
        "<summary>incorrect id format</summary></entry>"
    )
    with pytest.raises(ExternalServiceError, match="rejected"):
        parse_feed(atom_feed([error_entry]))


async def test_search_success_and_params(settings):
    calls: list[httpx.Request] = []
    result = await client(settings, arxiv_handler(calls)).search(
        "graph neural networks", 3, start_date=date(2022, 1, 1)
    )
    assert len(result.papers) == 3 and result.total_results == 1234 and result.message is None
    params = calls[0].url.params
    assert calls[0].url.host == "export.arxiv.org" and calls[0].url.scheme == "https"
    assert params["max_results"] == "3" and "submittedDate:[202201010000" in params["search_query"]


async def test_search_empty_results(settings):
    result = await client(settings, arxiv_handler()).search("nothingmatches")
    assert result.papers == [] and "No arXiv papers" in result.message


async def test_search_retries_on_503_then_succeeds(settings):
    responses = [httpx.Response(503), httpx.Response(200, text=atom_feed([atom_entry("1710.10903", "GAT")]))]
    result = await client(settings, lambda r: responses.pop(0)).search("gat")
    assert result.papers[0].title == "GAT"


async def test_search_fails_after_repeated_rate_limits(settings):
    with pytest.raises(ExternalServiceError, match="HTTP 429"):
        await client(settings, lambda r: httpx.Response(429)).search("gnn")


async def test_search_timeout(settings):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ExternalServiceError, match="timed out"):
        await client(settings, handler).search("gnn")


async def test_search_client_error_is_not_retried(settings):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400)

    with pytest.raises(ExternalServiceError, match="HTTP 400"):
        await client(settings, handler).search("gnn")
    assert len(calls) == 1


async def test_results_are_cached(settings):
    calls: list[httpx.Request] = []
    arxiv = client(settings, arxiv_handler(calls))
    await arxiv.search("gnn")
    await arxiv.search("gnn")
    assert len(calls) == 1


async def test_related_papers_excludes_seed(settings):
    calls: list[httpx.Request] = []
    result = await client(settings, arxiv_handler(calls)).related("1609.02907", max_results=5)
    assert result.related_to == "1609.02907v1"
    assert all(not p.arxiv_id.startswith("1609.02907") for p in result.papers)
    assert "ti:" in calls[1].url.params["search_query"]


async def test_related_unknown_paper(settings):
    result = await client(settings, arxiv_handler()).related("2501.00001")
    assert result.papers == [] and "no paper" in result.message
