"""
utils/log.py
Structured logging with ASCII-safe markers for Windows compatibility.
"""

import logging
import sys


_CONFIGURED = False


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that replaces unencodable characters instead of crashing."""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Fallback: encode with replace mode
                stream.write(msg.encode(stream.encoding or "utf-8", errors="replace").decode(stream.encoding or "utf-8", errors="replace") + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = SafeStreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Configures logging on first call."""
    setup_logging()
    return logging.getLogger(name)
