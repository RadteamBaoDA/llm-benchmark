from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(slots=True, kw_only=True)
class Event:
    """Base event type with timestamp metadata."""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True, kw_only=True)
class RequestEvent(Event):
    """Base class for request lifecycle events."""
    request_id: int
    endpoint: str
    payload: Dict[str, Any]


@dataclass(slots=True, kw_only=True)
class RequestSubmitted(RequestEvent):
    """Request was enqueued for execution."""


@dataclass(slots=True, kw_only=True)
class RequestStarted(RequestEvent):
    """Request execution began."""
    url: str


@dataclass(slots=True, kw_only=True)
class RequestCompleted(RequestEvent):
    """Request completed successfully."""
    latency: float
    status_code: int
    tokens: int
    prompt_tokens: int
    completion_tokens: int
    response: Optional[Dict[str, Any]] = None


@dataclass(slots=True, kw_only=True)
class RequestFailed(RequestEvent):
    """Request failed with an error."""
    latency: float
    status_code: Optional[int]
    error: str


@dataclass(slots=True, kw_only=True)
class ScenarioStarted(Event):
    """Benchmark scenario execution started."""
    scenario: str


@dataclass(slots=True, kw_only=True)
class ScenarioCompleted(Event):
    """Benchmark scenario execution finished."""
    scenario: str
    duration: float
    success: bool
    error: Optional[str] = None
