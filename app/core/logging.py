"""Logging setup with a redaction filter so secrets never reach log output."""

from __future__ import annotations

import logging
import re

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{10,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"(?i:bearer)\s+[A-Za-z0-9._\-]{8,}|(?i:x-api-key)[\"':\s=]+[A-Za-z0-9._\-]{8,})"
)


class RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if _SECRET_RE.search(message):
            record.msg = _SECRET_RE.sub("[REDACTED]", message)
            record.args = ()
        return True


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    redactor = RedactSecretsFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(redactor)
    # httpx logs full request URLs at INFO; keep them at WARNING.
    for noisy in ("httpx", "httpx2", "httpcore", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
