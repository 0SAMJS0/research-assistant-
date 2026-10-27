import logging

import pytest
from pydantic import ValidationError

from app.agent.grounding import check_citations
from app.core.config import Settings
from app.core.logging import RedactSecretsFilter
from app.core.security import RateLimiter, StaticTokenVerifier, safe_child, sanitize_filename


def test_production_fails_closed_without_auth():
    with pytest.raises(ValidationError, match="APP_API_KEY"):
        Settings(_env_file=None, app_env="production", app_api_key=None, mcp_auth_token="x" * 32)
    with pytest.raises(ValidationError, match="MCP_AUTH_TOKEN"):
        Settings(_env_file=None, app_env="production", app_api_key="x" * 32, mcp_auth_token="short")
    ok = Settings(_env_file=None, app_env="production", app_api_key="x" * 32, mcp_auth_token="y" * 32)
    assert ok.is_production


def test_chunk_overlap_validation():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=500, chunk_overlap=500)


def test_mcp_allowed_hosts_include_service_host():
    s = Settings(_env_file=None, allowed_hosts="research.example.com", mcp_base_url="http://mcp:8100/mcp")
    assert {"research.example.com", "research.example.com:*", "mcp", "mcp:*"} <= set(s.mcp_allowed_hosts)


def test_safe_child(tmp_path):
    assert safe_child(tmp_path, "a.csv") == (tmp_path / "a.csv").resolve()
    for bad in ["../a.csv", "..", "a/b.csv", "/etc/passwd", ".env", ""]:
        with pytest.raises(ValueError):
            safe_child(tmp_path, bad)


def test_sanitize_filename():
    assert sanitize_filename("../../etc/pass wd.pdf") == "pass_wd.pdf"
    assert sanitize_filename("...") == "upload"


def test_rate_limiter():
    rl = RateLimiter()
    assert all(rl.check("ip", "b", 3) for _ in range(3))
    assert not rl.check("ip", "b", 3)
    assert rl.check("other", "b", 3)


async def test_static_token_verifier():
    v = StaticTokenVerifier("s" * 20, "http://x/mcp")
    assert (await v.verify_token("s" * 20)).resource == "http://x/mcp"
    assert await v.verify_token("wrong") is None


def test_log_redaction():
    record = logging.LogRecord(
        "x", logging.INFO, "f", 1, "key %s and Bearer abcdefghijkl", ("sk-" + "a" * 30,), None
    )
    RedactSecretsFilter().filter(record)
    assert "sk-" not in record.getMessage() and "abcdefghijkl" not in record.getMessage()


EVIDENCE = {
    "query_documents": [{"passages": [{"filename": "gnn_primer.pdf", "page": 3}]}],
    "search_papers": [{"papers": [{"arxiv_id": "1609.02907v4"}]}],
    "analyze_repository": [{"repository": "octo/tiny-gnn"}],
    "generate_report": [{"report_id": "a" * 32, "references": [{"url": "https://arxiv.org/abs/1710.10903"}]}],
}


def test_citation_grounding_accepts_real_and_flags_invented():
    answer = (
        "Over-smoothing [gnn_primer.pdf, p. 3]; invented [gnn_primer.pdf, p. 9] and [other.pdf, p. 1]. "
        "GCN https://arxiv.org/abs/1609.02907 GAT arXiv:1710.10903, fake https://arxiv.org/abs/2502.00001. "
        "Code https://github.com/octo/tiny-gnn/blob/main/x.py and https://github.com/evil/repo. "
        f"Report /api/reports/{'a' * 32}?format=md and /api/reports/{'b' * 32}"
    )
    result = check_citations(answer, EVIDENCE)
    assert set(result["verified"]) == {
        "[gnn_primer.pdf, p. 3]",
        "arXiv:1609.02907",
        "arXiv:1710.10903",
        "github.com/octo/tiny-gnn",
        f"report {'a' * 32}",
    }
    assert set(result["unverified"]) == {
        "[gnn_primer.pdf, p. 9]",
        "[other.pdf, p. 1]",
        "arXiv:2502.00001",
        "github.com/evil/repo",
        f"report {'b' * 32}",
    }


def test_citation_grounding_without_evidence():
    assert check_citations("No sources here.", {}) == {"verified": [], "unverified": []}
