"""Read-only GitHub REST API repository analyzer. Never clones or executes repository code."""

from __future__ import annotations

import asyncio
import base64
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath

import httpx

from app.core.config import Settings
from app.models.tools import CodeReference, RepositoryAnalysis, RepositoryFile
from app.services.http import ExternalServiceError, build_client

GITHUB_HOSTS = {"api.github.com"}
OWNER_RE = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
REPO_RE = r"[A-Za-z0-9._-]{1,100}"
REPO_URL_RE = re.compile(
    rf"^(?:(?:https?://)?(?:www\.)?github\.com/)?(?P<owner>{OWNER_RE})/(?P<repo>{REPO_RE}?)(?:\.git)?(?:/.*)?$"
)

DEPENDENCY_FILES = {
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "environment.yml",
    "environment.yaml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "conda.yaml",
}
# Files that may hold credentials are never fetched, whatever their name suggests.
SECRET_PATTERNS = re.compile(
    r"(^|/)(\.env(\..*)?|.*\.(pem|key|p12|pfx|keystore|jks)|id_rsa.*|id_ed25519.*|.*secret.*|"
    r".*credential.*|.*password.*|\.npmrc|\.pypirc|\.netrc|.*token.*)$",
    re.IGNORECASE,
)
SOURCE_EXTENSIONS = {".py", ".ipynb", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".cc", ".c", ".jl", ".r"}
RELEVANT_NAME_HINTS = (
    "model",
    "train",
    "network",
    "layer",
    "main",
    "net",
    "attention",
    "transformer",
    "gnn",
    "graph",
    "conv",
    "loss",
    "optim",
    "trainer",
    "engine",
    "module",
    "encoder",
    "decoder",
    "core",
    "app",
)
ML_FRAMEWORKS = {
    "PyTorch": [r"\btorch\b"],
    "PyTorch Lightning": [r"pytorch[_-]lightning", r"\blightning\b"],
    "PyTorch Geometric": [r"torch[_-]geometric", r"\btorch_geometric\b"],
    "DGL": [r"\bdgl\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "Keras": [r"\bkeras\b"],
    "JAX": [r"\bjax\b"],
    "Flax": [r"\bflax\b"],
    "scikit-learn": [r"scikit-learn", r"\bsklearn\b"],
    "XGBoost": [r"\bxgboost\b"],
    "LightGBM": [r"\blightgbm\b"],
    "Hugging Face Transformers": [r"\btransformers\b"],
    "Hugging Face Datasets": [r"^datasets\b", r"from datasets import", r"import datasets\b"],
    "NumPy": [r"\bnumpy\b"],
    "pandas": [r"\bpandas\b"],
    "ONNX": [r"\bonnx(runtime)?\b"],
    "spaCy": [r"\bspacy\b"],
}
ALGORITHM_PATTERNS = {
    "Graph convolution (GCN)": r"\bGCN(Conv)?\b|graph convolution",
    "Graph attention (GAT)": r"\bGAT(Conv|v2Conv)?\b|graph attention",
    "GraphSAGE": r"\bSAGEConv\b|graphsage",
    "Message passing": r"MessagePassing|message passing",
    "Transformer / self-attention": r"MultiheadAttention|self[-_ ]attention|\bTransformer\b",
    "Convolutional network": r"\bConv[123]d\b|convolution",
    "Recurrent network (LSTM/GRU)": r"\b(LSTM|GRU)\b",
    "Diffusion model": r"diffusion|DDPM|noise schedule",
    "Variational autoencoder": r"\bVAE\b|variational autoencoder|reparameteri[sz]",
    "Contrastive learning": r"contrastive|InfoNCE|SimCLR",
    "Gradient boosting": r"XGB|LGBM|GradientBoosting",
    "Reinforcement learning": r"policy gradient|\bPPO\b|Q[-_ ]learning|replay buffer",
}
MAX_SELECTED_FILES = 8
MAX_FILE_BYTES = 60_000
EXCERPT_CHARS = 1_500


def parse_repository(value: str) -> tuple[str, str]:
    match = REPO_URL_RE.match(value.strip())
    if not match:
        raise ValueError("Repository must be 'owner/repo' or a https://github.com/owner/repo URL")
    owner, repo = match.group("owner"), match.group("repo")
    if repo in {".", ".."} or not repo:
        raise ValueError("Invalid repository name")
    return owner, repo


def _score_source(path: str) -> int:
    p = PurePosixPath(path)
    if p.suffix.lower() not in SOURCE_EXTENSIONS or SECRET_PATTERNS.search(path):
        return -1
    lowered = path.lower()
    if any(part in lowered for part in ("test", "docs/", "example", "vendor/", "third_party", "migrations")):
        score = 1
    else:
        score = 5
    score += sum(3 for hint in RELEVANT_NAME_HINTS if hint in p.stem.lower())
    score -= len(p.parts)  # prefer shallow files
    if p.suffix == ".py":
        score += 2
    return score


def detect_frameworks(texts: list[str]) -> list[str]:
    blob = "\n".join(texts)
    found = []
    for name, patterns in ML_FRAMEWORKS.items():
        if any(re.search(pat, blob, re.IGNORECASE | re.MULTILINE) for pat in patterns):
            found.append(name)
    return found


def extract_references(path: str, text: str, url: str) -> tuple[list[CodeReference], set[str]]:
    refs: list[CodeReference] = []
    algorithms: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        class_match = re.match(r"\s*class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", line)
        if class_match:
            bases = class_match.group(2)
            kind = (
                "model class"
                if re.search(r"Module|Model|Layer|LightningModule|MessagePassing", bases)
                else "class"
            )
            refs.append(
                CodeReference(
                    path=path, line=lineno, symbol=class_match.group(1), kind=kind, url=f"{url}#L{lineno}"
                )
            )
        fn_match = re.match(
            r"(?:async\s+)?def\s+((?:train|fit|forward|evaluate|main|predict|loss)\w*)\s*\(", line
        )
        if fn_match:
            refs.append(
                CodeReference(
                    path=path, line=lineno, symbol=fn_match.group(1), kind="function", url=f"{url}#L{lineno}"
                )
            )
        for algo, pattern in ALGORITHM_PATTERNS.items():
            if re.search(pattern, line, re.IGNORECASE):
                algorithms.add(algo)
    return refs[:25], algorithms


def parse_dependencies(name: str, text: str) -> list[str]:
    deps: list[str] = []
    if name.startswith("requirements") or name == "Pipfile":
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith(("-", "[")):
                deps.append(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0])
    elif name == "pyproject.toml" or name in {"setup.py", "setup.cfg"}:
        for match in re.finditer(
            r"""["']([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~][^"']*)?["']""", text
        ):
            token = match.group(1)
            if not re.fullmatch(r"\d[\d.]*|py\d+|[A-Z].*|.*\.(py|md|txt|toml)", token):
                deps.append(token)
    elif name == "package.json":
        deps.extend(re.findall(r'"([@a-z0-9_./\-]+)"\s*:\s*"[\^~>=<\d*]', text))
    elif name.startswith("environment"):
        deps.extend(m.group(1) for m in re.finditer(r"^\s*-\s*([A-Za-z0-9_.\-]+)", text, re.MULTILINE))
    return sorted({d.lower() for d in deps if len(d) > 1})[:80]


class GitHubClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        token = self._settings.secret("github_token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return build_client(
            self._settings.github_api_url,
            GITHUB_HOSTS,
            self._settings.http_timeout_s,
            headers,
            transport=self._transport,
        )

    async def _get(self, client: httpx.AsyncClient, path: str, **params: str) -> httpx.Response:
        try:
            response = await client.get(path, params=params or None)
        except httpx.TimeoutException as exc:
            raise ExternalServiceError("GitHub API request timed out") from exc
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"Network error contacting GitHub ({type(exc).__name__})") from exc
        remaining = response.headers.get("X-RateLimit-Remaining")
        if response.status_code in (403, 429) and (remaining == "0" or response.status_code == 429):
            reset = response.headers.get("X-RateLimit-Reset")
            when = (
                datetime.fromtimestamp(int(reset), tz=UTC).isoformat()
                if reset and reset.isdigit()
                else "later"
            )
            raise ExternalServiceError(
                f"GitHub API rate limit exhausted; resets at {when}. Set GITHUB_TOKEN for a higher quota."
            )
        if response.status_code == 401:
            raise ExternalServiceError("GitHub rejected the configured GITHUB_TOKEN (401)")
        return response

    async def analyze(self, repository: str, max_files: int = 6) -> RepositoryAnalysis:
        owner, repo = parse_repository(repository)
        max_files = max(1, min(max_files, MAX_SELECTED_FILES))
        notes: list[str] = []
        async with self._client() as client:
            meta_resp = await self._get(client, f"/repos/{owner}/{repo}")
            if meta_resp.status_code == 404:
                raise ExternalServiceError(f"Repository {owner}/{repo} was not found or is private")
            if meta_resp.status_code >= 400:
                raise ExternalServiceError(f"GitHub returned HTTP {meta_resp.status_code} for {owner}/{repo}")
            meta = meta_resp.json()
            full_name = meta["full_name"]
            branch = meta.get("default_branch") or "main"

            lang_resp, readme_resp, branch_resp = await asyncio.gather(
                self._get(client, f"/repos/{full_name}/languages"),
                self._get(client, f"/repos/{full_name}/readme"),
                self._get(client, f"/repos/{full_name}/branches/{branch}"),
            )
            languages_raw = lang_resp.json() if lang_resp.status_code == 200 else {}
            total_bytes = sum(languages_raw.values()) or 1
            languages = {k: round(v * 100 / total_bytes, 2) for k, v in languages_raw.items()}

            readme_text = None
            if readme_resp.status_code == 200:
                readme_text = self._decode(readme_resp.json())
            else:
                notes.append("Repository has no README.")

            sha = None
            if branch_resp.status_code == 200:
                sha = branch_resp.json().get("commit", {}).get("sha")
            tree_ref = sha or branch
            tree_resp = await self._get(client, f"/repos/{full_name}/git/trees/{tree_ref}", recursive="1")
            tree = tree_resp.json() if tree_resp.status_code == 200 else {"tree": [], "truncated": False}
            if tree_resp.status_code != 200:
                notes.append(f"File tree unavailable (HTTP {tree_resp.status_code}).")
            blobs = [e for e in tree.get("tree", []) if e.get("type") == "blob"]
            if tree.get("truncated"):
                notes.append("GitHub truncated the file tree (very large repository); analysis is partial.")

            top_level = sorted(
                {
                    PurePosixPath(e["path"]).parts[0] + ("/" if "/" in e["path"] else "")
                    for e in tree.get("tree", [])
                    if e.get("path")
                }
            )[:60]
            dep_entries = [
                e
                for e in blobs
                if PurePosixPath(e["path"]).name in DEPENDENCY_FILES
                and len(PurePosixPath(e["path"]).parts) <= 2
                and not SECRET_PATTERNS.search(e["path"])
            ]
            dep_entries = sorted(dep_entries, key=lambda e: len(e["path"]))[:4]
            source_entries = sorted(
                (e for e in blobs if _score_source(e["path"]) >= 0 and e.get("size", 0) <= MAX_FILE_BYTES),
                key=lambda e: (-_score_source(e["path"]), e["path"]),
            )[:max_files]

            ref = sha or branch
            blob_base = f"https://github.com/{full_name}/blob/{ref}"
            fetched = await asyncio.gather(
                *(self._fetch_file(client, full_name, e["path"], ref) for e in dep_entries + source_entries)
            )

        contents = dict(zip([e["path"] for e in dep_entries + source_entries], fetched, strict=True))
        dependencies: list[str] = []
        for entry in dep_entries:
            text = contents.get(entry["path"])
            if text:
                dependencies.extend(parse_dependencies(PurePosixPath(entry["path"]).name, text))
        dependencies = sorted(set(dependencies))

        code_refs: list[CodeReference] = []
        algorithms: set[str] = set()
        selected: list[RepositoryFile] = []
        for entry in source_entries:
            text = contents.get(entry["path"])
            if text is None:
                continue
            url = f"{blob_base}/{entry['path']}"
            refs, algos = extract_references(entry["path"], text, url)
            code_refs.extend(refs)
            algorithms |= algos
            selected.append(
                RepositoryFile(
                    path=entry["path"], size=entry.get("size", 0), url=url, excerpt=text[:EXCERPT_CHARS]
                )
            )

        framework_texts = [t for t in contents.values() if t] + [readme_text or ""]
        frameworks = detect_frameworks(framework_texts)
        summary = self._summarize(meta, languages, top_level, blobs, frameworks, dep_entries)

        return RepositoryAnalysis(
            repository=full_name,
            url=meta.get("html_url", f"https://github.com/{full_name}"),
            description=meta.get("description"),
            default_branch=branch,
            commit_sha=sha,
            stars=meta.get("stargazers_count", 0),
            forks=meta.get("forks_count", 0),
            open_issues=meta.get("open_issues_count", 0),
            license=(meta.get("license") or {}).get("spdx_id"),
            topics=meta.get("topics", []),
            created_at=meta.get("created_at"),
            pushed_at=meta.get("pushed_at"),
            languages=languages,
            readme_excerpt=readme_text[:3000] if readme_text else None,
            file_count=len(blobs),
            tree_truncated=bool(tree.get("truncated")),
            top_level=top_level,
            dependency_files=[e["path"] for e in dep_entries],
            dependencies=dependencies,
            ml_frameworks=frameworks,
            key_algorithms=sorted(algorithms),
            architecture_summary=summary,
            selected_files=selected,
            code_references=code_refs[:60],
            rate_limit_remaining=self._remaining(meta_resp),
            notes=notes,
        )

    @staticmethod
    def _remaining(response: httpx.Response) -> int | None:
        value = response.headers.get("X-RateLimit-Remaining")
        return int(value) if value and value.isdigit() else None

    @staticmethod
    def _decode(payload: dict) -> str | None:
        if payload.get("encoding") != "base64" or "content" not in payload:
            return None
        try:
            return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return None

    async def _fetch_file(self, client: httpx.AsyncClient, full_name: str, path: str, ref: str) -> str | None:
        if SECRET_PATTERNS.search(path):
            return None
        response = await self._get(client, f"/repos/{full_name}/contents/{path}", ref=ref)
        if response.status_code != 200:
            return None
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("size", 0) > MAX_FILE_BYTES:
            return None
        return self._decode(payload)

    @staticmethod
    def _summarize(
        meta: dict,
        languages: dict[str, float],
        top_level: list[str],
        blobs: list[dict],
        frameworks: list[str],
        dep_entries: list[dict],
    ) -> list[str]:
        paths = [e["path"] for e in blobs]
        lines = []
        if languages:
            primary = max(languages, key=lambda k: languages[k])
            lines.append(f"Primary language: {primary} ({languages[primary]}% of code bytes).")
        dirs = [p for p in top_level if p.endswith("/")]
        if dirs:
            lines.append("Top-level directories: " + ", ".join(dirs[:15]) + ".")
        test_files = [p for p in paths if re.search(r"(^|/)(tests?|spec)(/|_)|_test\.py$|test_.*\.py$", p)]
        lines.append(f"Test files detected: {len(test_files)}." if test_files else "No test files detected.")
        if any(p.startswith(".github/workflows/") for p in paths):
            lines.append("Continuous integration: GitHub Actions workflows present.")
        if any(PurePosixPath(p).name in {"Dockerfile", "docker-compose.yml", "compose.yaml"} for p in paths):
            lines.append("Containerization: Docker configuration present.")
        notebooks = [p for p in paths if p.endswith(".ipynb")]
        if notebooks:
            lines.append(f"Jupyter notebooks: {len(notebooks)}.")
        entrypoints = [
            p
            for p in paths
            if PurePosixPath(p).name in {"main.py", "train.py", "app.py", "run.py", "__main__.py", "cli.py"}
        ]
        if entrypoints:
            lines.append("Likely entry points: " + ", ".join(entrypoints[:8]) + ".")
        if dep_entries:
            lines.append("Dependency manifests: " + ", ".join(e["path"] for e in dep_entries) + ".")
        if frameworks:
            lines.append("Detected ML/data frameworks: " + ", ".join(frameworks) + ".")
        return lines
