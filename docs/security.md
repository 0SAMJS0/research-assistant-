# Security model

| Threat | Mitigation |
|---|---|
| Unauthenticated access to private data | All `/api/*` routes (except `/api/config`) need `APP_API_KEY`. The key is checked in middleware *before* body parsing, with a constant-time compare. `/mcp` needs `MCP_AUTH_TOKEN` through the MCP SDK's bearer middleware and `TokenVerifier`, with resource validation. |
| Misconfigured production | `APP_ENV=production` refuses to start without both secrets (at least 16 characters and not the placeholder) and without a Chroma server. |
| Secret leakage | No secrets in code or images. `.env` is git- and docker-ignored. A log filter redacts `sk-…`, GitHub tokens, Bearer values and `X-API-Key` values. httpx URL logging is lowered to WARNING. Secrets are held as `SecretStr`. |
| SSRF | Outbound HTTP only goes through `build_client`, which allows `https` to `export.arxiv.org`/`arxiv.org` and `api.github.com` only. A request hook re-checks every request, including redirects. Repository identifiers are regex-validated, and arXiv IDs are normalized and validated. |
| Path traversal / arbitrary file access | Experiment and report files are resolved by `safe_child` (strict name regex, resolved parent must equal the base directory). Report IDs must be 32 hex characters. Upload names are sanitized. |
| Malicious uploads | Extension, content type, size (streamed, capped at `MAX_UPLOAD_MB`) and content checks: `%PDF-` magic, pypdf parse, page cap, encryption rejected, UTF-8 CSV/JSON parsed by pandas or json only. Files are never executed. |
| Prompt injection via PDFs, READMEs or abstracts | The system prompt and MCP server instructions mark tool output as untrusted data. The report tool accepts no facts from the model. The citation grounding check flags invented references. Untrusted text is Markdown-escaped in reports, and the HTML renderer (`markdown-it`, `html=False`) drops raw HTML. |
| XSS | The UI escapes HTML before its small Markdown renderer and only links `http(s)` and `/api/` URLs. A strict CSP (`script-src 'self'`, no inline script) applies. Report HTML is served with `default-src 'none'`. `X-Frame-Options: DENY` and `nosniff` are set. |
| DNS rebinding / cross-origin MCP use | MCP `TransportSecuritySettings` validates `Host` against `ALLOWED_HOSTS` (421) and `Origin` against `ALLOWED_ORIGINS` (403). FastAPI uses `TrustedHostMiddleware` and a strict CORS allow-list. |
| Abuse and resource exhaustion | Per-IP rate limits (separate, lower limit for chat). Agent tool budget, recursion limit, per-tool and per-request timeouts. HTTP timeouts on all outbound calls. MCP request body cap of 1 MB. arXiv pacing and caching. GitHub fetches capped at 8 files of 60 KB or less. |
| GitHub secrets | Paths that look like credentials (`.env*`, `*.pem`, `*key*`, `*secret*`, `*token*`, `id_rsa*`, …) are never fetched. Repository code is never cloned or executed. |
| Container breakout / lateral movement | Non-root UID 10001, read-only root filesystem, `cap_drop: ALL`, `no-new-privileges`. ChromaDB and the standalone MCP service are reachable only on the internal network. |
| Cache leakage | API responses are sent with `Cache-Control: no-store`. |

## Residual risks

- A single shared API key is not per-user authorization. Put an identity-aware proxy (OIDC) in front for
  multi-user deployments.
- An LLM can still misstate evidence in prose. Citations are verified mechanically, but semantic faithfulness is
  not. Reports keep model interpretation separate from observed data for this reason.
- Rate limiting is in-process.
