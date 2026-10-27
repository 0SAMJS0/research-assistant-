"""Application settings loaded from environment variables (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Runtime
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # LLM
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    openai_base_url: str | None = None
    llm_timeout_s: float = 60.0

    # Embeddings: "local" = Chroma's bundled ONNX all-MiniLM-L6-v2, "openai" = OpenAI embeddings,
    # "hash" = deterministic hashing embeddings (tests/offline demos only).
    embedding_provider: Literal["local", "openai", "hash"] = "local"
    openai_embedding_model: str = "text-embedding-3-small"

    # Integrations
    github_token: SecretStr | None = None
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    github_api_url: str = "https://api.github.com"
    http_timeout_s: float = 20.0
    arxiv_min_interval_s: float = 3.0

    # Auth
    app_api_key: SecretStr | None = None
    mcp_auth_token: SecretStr | None = None

    # URLs
    public_base_url: str = "http://localhost:8000"
    mcp_base_url: str = "http://localhost:8000/mcp"

    # ChromaDB: "http" uses a Chroma server (production); "ephemeral" is in-memory (tests only).
    chroma_mode: Literal["http", "ephemeral"] = "http"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_ssl: bool = False

    # Storage
    data_dir: Path = Path("./data/runtime")
    reports_dir: Path = Path("./data/reports")

    # HTTP hardening
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    allowed_hosts: str = "localhost,127.0.0.1"
    max_upload_mb: int = Field(default=20, ge=1, le=200)
    rate_limit_per_minute: int = Field(default=60, ge=1)
    chat_rate_limit_per_minute: int = Field(default=12, ge=1)

    # RAG
    chunk_size: int = Field(default=1000, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_max_distance: float = Field(default=0.75, gt=0, le=2)

    # Agent limits
    agent_max_tool_calls: int = Field(default=8, ge=1, le=50)
    agent_recursion_limit: int = Field(default=25, ge=4, le=200)
    agent_request_timeout_s: float = Field(default=180.0, ge=5)
    agent_tool_timeout_s: float = Field(default=90.0, ge=1)
    agent_history_messages: int = Field(default=40, ge=4)

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.is_production:
            missing = [
                name
                for name, value in (
                    ("APP_API_KEY", self.app_api_key),
                    ("MCP_AUTH_TOKEN", self.mcp_auth_token),
                )
                if value is None
                or len(value.get_secret_value()) < 16
                or value.get_secret_value().lower().startswith(("change-me", "changeme"))
            ]
            if missing:
                # Fail closed: production never runs with authentication disabled.
                raise ValueError(
                    f"APP_ENV=production requires {', '.join(missing)} "
                    "(at least 16 random characters each, not the placeholder)"
                )
            if self.chroma_mode != "http":
                raise ValueError("APP_ENV=production requires CHROMA_MODE=http")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def origins(self) -> list[str]:
        return _split_csv(self.allowed_origins)

    @property
    def hosts(self) -> list[str]:
        return _split_csv(self.allowed_hosts)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @property
    def mcp_allowed_hosts(self) -> list[str]:
        """Host header values accepted by the MCP transport (DNS-rebinding protection)."""
        hosts: list[str] = []
        for host in [*self.hosts, urlparse(self.mcp_base_url).hostname or ""]:
            if host and host != "*":
                hosts.extend([host, f"{host}:*"])
        return sorted(set(hosts))

    def secret(self, name: str) -> str | None:
        value = getattr(self, name)
        return value.get_secret_value() if value is not None else None

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.experiments_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
