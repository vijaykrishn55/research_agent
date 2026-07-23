"""
utils/events.py
Structured event collector for the research pipeline.

Records timestamped events at every pipeline stage so that a human-readable
research log can be rendered in the CLI (--verbose) or returned via the API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PipelineEvent:
    """A single timestamped event in the research pipeline."""

    timestamp: str          # HH:MM:SS.mmm
    phase: str              # pipeline, planner, search, retrieval, generation
    message: str            # Human-readable summary
    duration_ms: float | None = None
    details: dict = field(default_factory=dict)
    level: str = "info"     # info | warn | error


class ResearchLog:
    """
    Thread-local event collector for a single pipeline run.

    Usage:
        research_log.reset()          # at the start of a query
        research_log.emit(...)        # at each stage
        events = research_log.events  # at the end
    """

    def __init__(self) -> None:
        self._events: list[PipelineEvent] = []
        self._t0: float | None = None

    def reset(self) -> None:
        """Clear events and restart the wall-clock timer."""
        self._events.clear()
        self._t0 = time.time()

    def emit(
        self,
        phase: str,
        message: str,
        duration_ms: float | None = None,
        details: dict | None = None,
        level: str = "info",
    ) -> None:
        """Record a pipeline event."""
        self._events.append(PipelineEvent(
            timestamp=datetime.now().strftime("%H:%M:%S.") +
                      f"{datetime.now().microsecond // 1000:03d}",
            phase=phase,
            message=message,
            duration_ms=duration_ms,
            details=details or {},
            level=level,
        ))

    @property
    def events(self) -> list[PipelineEvent]:
        """Return a copy of all recorded events."""
        return list(self._events)

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock time since reset, in milliseconds."""
        if self._t0 is None:
            return 0.0
        return round((time.time() - self._t0) * 1000, 1)

    def to_dicts(self) -> list[dict]:
        """Serialize events to a JSON-friendly list of dicts."""
        return [
            {
                "timestamp": e.timestamp,
                "phase": e.phase,
                "message": e.message,
                "duration_ms": e.duration_ms,
                "details": e.details,
                "level": e.level,
            }
            for e in self._events
        ]


# Module-level singleton — one log per process (fine for CLI; API callers
# should reset() at the start of each request).
research_log = ResearchLog()
