import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable, DefaultDict, Dict, List, Type

from .events import Event

EventHandler = Callable[[Event], Any]


class EventBus:
    """Lightweight pub/sub event bus for benchmark lifecycle events."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[Type[Event], List[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        """Remove a handler if registered."""
        if handler in self._handlers.get(event_type, []):
            self._handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers."""
        for handler in list(self._handlers.get(type(event), [])):
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    def clear(self) -> None:
        """Remove all handlers (useful for tests)."""
        self._handlers.clear()
