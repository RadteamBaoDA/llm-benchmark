"""
Debug logging module for benchmark execution tracing using structlog.

Provides structured logging for:
- Execution mode lifecycle
- Request timing and concurrency
- Queue behavior analysis
- Error tracking and diagnostics

Uses structlog for powerful, flexible structured logging with:
- JSON output for production
- Colorful console output for development
- Context binding for request tracking
- Async support
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Union

import structlog
from structlog.typing import FilteringBoundLogger
from rich.console import Console
from rich.logging import RichHandler


# =============================================================================
# Custom Processors
# =============================================================================

def add_elapsed_time(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add elapsed time since benchmark start to log entries."""
    start_time = event_dict.pop("_start_time", None)
    if start_time is not None:
        elapsed = time.perf_counter() - start_time
        event_dict["elapsed"] = f"+{elapsed:.3f}s"
    return event_dict


def format_metrics(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Format numeric metrics with appropriate precision."""
    for key, value in list(event_dict.items()):
        if key.endswith("_ms") and isinstance(value, (int, float)):
            event_dict[key] = f"{value:.2f}"
        elif key.endswith("_sec") and isinstance(value, (int, float)):
            event_dict[key] = f"{value:.3f}"
    return event_dict


def add_benchmark_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add benchmark context from context variables if available."""
    return event_dict


# =============================================================================
# Structlog Configuration
# =============================================================================

def configure_structlog(
    console_output: bool = False,
    file_output: bool = True,
    log_file: str = "debug.log",
    level: int = logging.DEBUG,
    json_format: bool = False,
    colors: bool = True,
) -> None:
    """
    Configure structlog for benchmark logging.
    
    Args:
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        log_file: Path to log file
        level: Logging level
        json_format: Whether to use JSON format (for production)
        colors: Whether to use colored output (console only)
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Shared processors for both structlog and stdlib
    shared_processors: List[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%H:%M:%S.%f", utc=False),
        add_elapsed_time,
        format_metrics,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Determine renderer based on settings
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        # For RichHandler, we don't need a colored console renderer in the processor chain
        # because Rich handles the rendering. We just need to prep it for stdlib logging.
        renderer = structlog.dev.ConsoleRenderer(colors=False)
    
    # Create formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    # Setup handlers
    handlers = []
    
    if file_output:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        # Use non-colored renderer for file output
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer(colors=False),
            ],
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    if console_output:
        # Use RichHandler for console output
        # We need to ensure we don't have double timestamp/level since Rich does it
        console_handler = RichHandler(
            console=Console(force_terminal=True, color_system="truecolor"),
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )
        # Note: RichHandler formats the message itself, but we still need to pass 
        # the structlog-processed dict as the message.
        # However, RichHandler expects a string message. 
        # structlog.stdlib.ProcessorFormatter will render it to a string.
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        handlers.append(console_handler)
    
    # Configure root logger
    root_logger.setLevel(level)
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Prevent propagation of benchmark logs
    benchmark_logger = logging.getLogger("llm_benchmark")
    benchmark_logger.setLevel(level)
    benchmark_logger.propagate = False
    for handler in handlers:
        benchmark_logger.addHandler(handler)


# =============================================================================
# Debug Logger Class
# =============================================================================

class DebugLogger:
    """
    Structured debug logger for benchmark execution tracing using structlog.
    
    Provides rich context binding and structured logging for:
    - Mode lifecycle tracking
    - Request execution tracing
    - Concurrency monitoring
    - Queue behavior analysis
    """
    
    def __init__(
        self, 
        name: str = "llm_benchmark", 
        level: int = logging.DEBUG,
        console_output: bool = False, 
        file_output: bool = True,
        log_file: str = "debug.log",
        json_format: bool = False,
    ):
        self.name = name
        self.console_output = console_output
        self.file_output = file_output
        self.log_file = log_file
        self.json_format = json_format
        self._level = level
        
        # Configure structlog
        configure_structlog(
            console_output=console_output,
            file_output=file_output,
            log_file=log_file,
            level=level,
            json_format=json_format,
        )
        
        # Get the structlog logger
        self._logger: FilteringBoundLogger = structlog.get_logger(name)
        
        # Benchmark state
        self._start_time: Optional[float] = None
        self._request_count: int = 0
        self._mode: Optional[str] = None
    
    @property
    def logger(self) -> FilteringBoundLogger:
        """Get the underlying structlog logger."""
        return self._logger
    
    def bind(self, **kwargs) -> "DebugLogger":
        """Bind context to the logger, returning a new logger with that context."""
        new_logger = DebugLogger.__new__(DebugLogger)
        new_logger.name = self.name
        new_logger.console_output = self.console_output
        new_logger.file_output = self.file_output
        new_logger.log_file = self.log_file
        new_logger.json_format = self.json_format
        new_logger._level = self._level
        new_logger._start_time = self._start_time
        new_logger._request_count = self._request_count
        new_logger._mode = self._mode
        new_logger._logger = self._logger.bind(**kwargs)
        return new_logger
    
    def unbind(self, *keys: str) -> "DebugLogger":
        """Remove keys from the logger context."""
        new_logger = DebugLogger.__new__(DebugLogger)
        new_logger.name = self.name
        new_logger.console_output = self.console_output
        new_logger.file_output = self.file_output
        new_logger.log_file = self.log_file
        new_logger.json_format = self.json_format
        new_logger._level = self._level
        new_logger._start_time = self._start_time
        new_logger._request_count = self._request_count
        new_logger._mode = self._mode
        new_logger._logger = self._logger.unbind(*keys)
        return new_logger
    
    def _with_elapsed(self) -> Dict[str, Any]:
        """Get context dict with elapsed time."""
        if self._start_time is not None:
            return {"_start_time": self._start_time}
        return {}
    
    def enable_console(self) -> None:
        """Enable console output."""
        if not self.console_output:
            self.console_output = True
            configure_structlog(
                console_output=True,
                file_output=self.file_output,
                log_file=self.log_file,
                level=self._level,
                json_format=self.json_format,
            )
            self._logger = structlog.get_logger(self.name)
    
    def set_level(self, level: int) -> None:
        """Change logging level."""
        self._level = level
        logging.getLogger(self.name).setLevel(level)
        logging.getLogger().setLevel(level)
    
    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return self._level <= logging.DEBUG
    
    # -------------------------------------------------------------------------
    # Mode Lifecycle Logging
    # -------------------------------------------------------------------------
    
    def mode_start(self, mode: str, scenario_name: str, config: Dict[str, Any]) -> None:
        """Log execution mode start with structured context."""
        self._start_time = time.perf_counter()
        self._mode = mode
        self._request_count = 0
        
        # Clear any previous context and bind new mode context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            mode=mode,
            scenario=scenario_name,
        )
        
        self._logger.info(
            "mode_start",
            mode=mode.upper(),
            scenario=scenario_name,
            config=config,
            **self._with_elapsed(),
        )
    
    def mode_phase(self, phase: str, details: str = "") -> None:
        """Log mode phase transition."""
        self._logger.info(
            "mode_phase",
            phase=phase,
            details=details if details else None,
            **self._with_elapsed(),
        )
    
    def mode_end(self, stats: Dict[str, Any]) -> None:
        """Log execution mode completion with final stats."""
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        
        self._logger.info(
            "mode_end",
            mode=self._mode,
            duration_sec=elapsed,
            stats=stats,
            **self._with_elapsed(),
        )
        
        # Clear context
        structlog.contextvars.clear_contextvars()
    
    # -------------------------------------------------------------------------
    # Request Lifecycle Logging
    # -------------------------------------------------------------------------
    
    def request_submit(self, request_id: int, concurrent: int) -> None:
        """Log request submission."""
        self._request_count += 1
        self._logger.debug(
            "request_submit",
            request_id=request_id,
            concurrent=concurrent,
            total_submitted=self._request_count,
            **self._with_elapsed(),
        )
    
    def request_start(self, request_id: int, wait_time_ms: float, concurrent: int) -> None:
        """Log request execution start."""
        self._logger.debug(
            "request_start",
            request_id=request_id,
            wait_ms=wait_time_ms,
            concurrent=concurrent,
            **self._with_elapsed(),
        )
    
    def request_complete(
        self, 
        request_id: int, 
        latency_ms: float, 
        tokens: int, 
        success: bool, 
        concurrent: int
    ) -> None:
        """Log request completion."""
        log_method = self._logger.debug if success else self._logger.warning
        log_method(
            "request_complete",
            request_id=request_id,
            latency_ms=latency_ms,
            tokens=tokens,
            success=success,
            concurrent=concurrent,
            **self._with_elapsed(),
        )
    
    def request_error(self, request_id: int, error: str, latency_ms: float) -> None:
        """Log request error."""
        self._logger.warning(
            "request_error",
            request_id=request_id,
            error=error,
            latency_ms=latency_ms,
            **self._with_elapsed(),
        )
    
    # -------------------------------------------------------------------------
    # Concurrency and Queue Logging
    # -------------------------------------------------------------------------
    
    def concurrency_change(self, active: int, peak: int, target: int) -> None:
        """Log concurrency level change."""
        self._logger.debug(
            "concurrency_change",
            active=active,
            peak=peak,
            target=target,
            **self._with_elapsed(),
        )
    
    def queue_sample(self, depth: int, wait_time_ms: float) -> None:
        """Log queue depth sample."""
        self._logger.debug(
            "queue_sample",
            depth=depth,
            wait_ms=wait_time_ms,
            **self._with_elapsed(),
        )
    
    def rate_limit(self, target_rps: float, actual_rps: float, adjustment_ms: float) -> None:
        """Log rate limiting adjustment."""
        self._logger.debug(
            "rate_limit",
            target_rps=target_rps,
            actual_rps=actual_rps,
            adjustment_ms=adjustment_ms,
            **self._with_elapsed(),
        )
    
    # -------------------------------------------------------------------------
    # Step/Stage Logging
    # -------------------------------------------------------------------------
    
    def step_start(
        self, 
        step: int, 
        total_steps: int, 
        concurrency: int, 
        requests: int, 
        duration: float = 0
    ) -> None:
        """Log step start in stepping/ramp-up modes."""
        self._logger.info(
            "step_start",
            step=step,
            total_steps=total_steps,
            concurrency=concurrency,
            requests=requests,
            duration_sec=duration,
            **self._with_elapsed(),
        )
    
    def step_complete(self, step: int, total_steps: int, completed: int, elapsed: float) -> None:
        """Log step completion."""
        self._logger.info(
            "step_complete",
            step=step,
            total_steps=total_steps,
            completed=completed,
            elapsed_sec=elapsed,
            **self._with_elapsed(),
        )
    
    def stage_start(
        self, 
        stage: int, 
        total_stages: int, 
        start_threads: int, 
        end_threads: int, 
        duration: float
    ) -> None:
        """Log ultimate thread group stage start."""
        if end_threads > start_threads:
            direction = "ramp_up"
        elif end_threads < start_threads:
            direction = "ramp_down"
        else:
            direction = "hold"
        
        self._logger.info(
            "stage_start",
            stage=stage,
            total_stages=total_stages,
            start_threads=start_threads,
            end_threads=end_threads,
            direction=direction,
            duration_sec=duration,
            **self._with_elapsed(),
        )
    
    # -------------------------------------------------------------------------
    # Spike Mode Logging
    # -------------------------------------------------------------------------
    
    def spike_phase(self, phase: str, concurrency: int, duration: float = 0) -> None:
        """Log spike test phase."""
        self._logger.info(
            "spike_phase",
            phase=phase.upper(),
            concurrency=concurrency,
            duration_sec=duration if duration > 0 else None,
            **self._with_elapsed(),
        )
    
    # -------------------------------------------------------------------------
    # Duration Mode Logging
    # -------------------------------------------------------------------------
    
    def duration_progress(
        self, 
        elapsed: float, 
        target: float, 
        requests_sent: int, 
        active: int
    ) -> None:
        """Log duration mode progress."""
        remaining = target - elapsed
        progress_pct = (elapsed / target) * 100 if target > 0 else 0
        
        self._logger.debug(
            "duration_progress",
            elapsed_sec=elapsed,
            remaining_sec=remaining,
            progress_pct=progress_pct,
            requests_sent=requests_sent,
            active=active,
            **self._with_elapsed(),
        )
    
    # -------------------------------------------------------------------------
    # Warmup Logging
    # -------------------------------------------------------------------------
    
    def warmup_start(self, count: int) -> None:
        """Log warmup phase start."""
        self._logger.info(
            "warmup_start",
            count=count,
            **self._with_elapsed(),
        )
    
    def warmup_request(self, request_num: int, latency_ms: float, success: bool) -> None:
        """Log warmup request."""
        self._logger.debug(
            "warmup_request",
            request_num=request_num,
            latency_ms=latency_ms,
            success=success,
            **self._with_elapsed(),
        )
    
    def warmup_complete(self, count: int, total_time_ms: float) -> None:
        """Log warmup completion."""
        self._logger.info(
            "warmup_complete",
            count=count,
            total_ms=total_time_ms,
            **self._with_elapsed(),
        )
    
    # -------------------------------------------------------------------------
    # Summary Logging
    # -------------------------------------------------------------------------
    
    def summary(self, title: str, metrics: Dict[str, Any]) -> None:
        """Log summary with metrics."""
        self._logger.info(
            "summary",
            title=title,
            metrics=metrics,
            **self._with_elapsed(),
        )
    
    # -------------------------------------------------------------------------
    # Generic Logging Methods (for flexibility)
    # -------------------------------------------------------------------------
    
    def debug(self, event: str, **kwargs) -> None:
        """Log a debug message with structured context."""
        self._logger.debug(event, **kwargs, **self._with_elapsed())
    
    def info(self, event: str, **kwargs) -> None:
        """Log an info message with structured context."""
        self._logger.info(event, **kwargs, **self._with_elapsed())
    
    def warning(self, event: str, **kwargs) -> None:
        """Log a warning message with structured context."""
        self._logger.warning(event, **kwargs, **self._with_elapsed())
    
    def error(self, event: str, **kwargs) -> None:
        """Log an error message with structured context."""
        self._logger.error(event, **kwargs, **self._with_elapsed())
    
    def exception(self, event: str, **kwargs) -> None:
        """Log an exception with structured context and traceback."""
        self._logger.exception(event, **kwargs, **self._with_elapsed())


# =============================================================================
# Global Logger Management
# =============================================================================

_debug_logger: Optional[DebugLogger] = None


def get_debug_logger(
    console_output: bool = False,
    json_format: bool = False,
) -> DebugLogger:
    """Get or create the global debug logger."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(
            console_output=console_output,
            json_format=json_format,
        )
    return _debug_logger


def reset_debug_logger(
    console_output: bool = False,
    json_format: bool = False,
) -> DebugLogger:
    """Reset the debug logger with new settings."""
    global _debug_logger
    _debug_logger = DebugLogger(
        console_output=console_output,
        json_format=json_format,
    )
    return _debug_logger


def enable_debug_logging(
    level: int = logging.DEBUG,
    console_output: bool = False,
    json_format: bool = False,
) -> None:
    """Enable debug logging at specified level."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(
            console_output=console_output,
            json_format=json_format,
        )
    _debug_logger.set_level(level)
    if console_output and not _debug_logger.console_output:
        _debug_logger.enable_console()


def disable_debug_logging() -> None:
    """Disable debug logging."""
    logger = get_debug_logger()
    logger.set_level(logging.WARNING)


def get_structlog_logger(name: str = "llm_benchmark") -> FilteringBoundLogger:
    """
    Get a structlog logger for use in other modules.
    
    This is useful when you want to use structlog directly in other parts
    of the application while maintaining consistent configuration.
    
    Example:
        from llm_benchmark.debug_logger import get_structlog_logger
        
        log = get_structlog_logger(__name__)
        log.info("processing", item_id=123, status="started")
    """
    return structlog.get_logger(name)


def bind_contextvars(**kwargs) -> None:
    """
    Bind context variables that will be included in all log entries.
    
    This is useful for adding request-scoped or task-scoped context.
    
    Example:
        bind_contextvars(request_id="abc-123", user_id="user-456")
        log.info("processing")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_contextvars() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def unbind_contextvars(*keys: str) -> None:
    """Unbind specific context variables."""
    structlog.contextvars.unbind_contextvars(*keys)
