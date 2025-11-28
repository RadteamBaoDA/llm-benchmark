"""
Debug logging module for benchmark execution tracing.

Provides structured logging for:
- Execution mode lifecycle
- Request timing and concurrency
- Queue behavior analysis
- Error tracking and diagnostics
"""

import logging
import sys
import time
from typing import Optional


class DebugLogger:
    """
    Structured debug logger for benchmark execution tracing.
    
    Supports output to:
    - Console (stdout) via --debug-console
    - File (debug.log) by default with --debug
    """
    
    def __init__(
        self, 
        name: str = "llm_benchmark", 
        level: int = logging.DEBUG,
        console_output: bool = False, 
        file_output: bool = True,
        log_file: str = "debug.log"
    ):
        self.logger = logging.getLogger(name)
        self.console_output = console_output
        self.file_output = file_output
        self.log_file = log_file
        self._setup_logger(level)
        self._start_time: Optional[float] = None
        self._request_count = 0
        self._mode: Optional[str] = None
        
    def _setup_logger(self, level: int):
        """Configure logger with formatted output."""
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        if self.file_output:
            file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.setLevel(level)
        self.logger.propagate = False
    
    def enable_console(self):
        """Enable console output."""
        if not self.console_output:
            self.console_output = True
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def set_level(self, level: int):
        """Change logging level."""
        self.logger.setLevel(level)
    
    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return self.logger.isEnabledFor(logging.DEBUG)
    
    def _elapsed(self) -> str:
        """Get elapsed time since start."""
        if self._start_time:
            elapsed = time.perf_counter() - self._start_time
            return f"[+{elapsed:8.3f}s]"
        return "[+    0.000s]"
    
    # -------------------------------------------------------------------------
    # Mode Lifecycle Logging
    # -------------------------------------------------------------------------
    
    def mode_start(self, mode: str, scenario_name: str, config: dict):
        """Log execution mode start."""
        self._start_time = time.perf_counter()
        self._mode = mode
        self._request_count = 0
        self.logger.info(f"{'='*70}")
        self.logger.info(f"MODE START: {mode.upper()}")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Scenario: {scenario_name}")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.debug(f"{self._elapsed()} Mode initialization complete")
    
    def mode_phase(self, phase: str, details: str = ""):
        """Log mode phase transition."""
        msg = f"{self._elapsed()} PHASE: {phase}"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)
    
    def mode_end(self, stats: dict):
        """Log execution mode completion."""
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        self.logger.info(f"{self._elapsed()} MODE END: {self._mode}")
        self.logger.info(f"  Total Duration: {elapsed:.3f}s")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info(f"{'='*70}")
    
    # -------------------------------------------------------------------------
    # Request Lifecycle Logging
    # -------------------------------------------------------------------------
    
    def request_submit(self, request_id: int, concurrent: int):
        """Log request submission."""
        self._request_count += 1
        self.logger.debug(
            f"{self._elapsed()} REQ #{request_id:4d} SUBMIT | "
            f"concurrent={concurrent} | total_submitted={self._request_count}"
        )
    
    def request_start(self, request_id: int, wait_time_ms: float, concurrent: int):
        """Log request execution start."""
        self.logger.debug(
            f"{self._elapsed()} REQ #{request_id:4d} START  | "
            f"wait={wait_time_ms:7.2f}ms | concurrent={concurrent}"
        )
    
    def request_complete(
        self, 
        request_id: int, 
        latency_ms: float, 
        tokens: int, 
        success: bool, 
        concurrent: int
    ):
        """Log request completion."""
        status = "OK" if success else "FAIL"
        self.logger.debug(
            f"{self._elapsed()} REQ #{request_id:4d} {status:4s}   | "
            f"latency={latency_ms:7.2f}ms | tokens={tokens:4d} | concurrent={concurrent}"
        )
    
    def request_error(self, request_id: int, error: str, latency_ms: float):
        """Log request error."""
        self.logger.warning(
            f"{self._elapsed()} REQ #{request_id:4d} ERROR  | "
            f"latency={latency_ms:7.2f}ms | error={error}"
        )
    
    # -------------------------------------------------------------------------
    # Concurrency and Queue Logging
    # -------------------------------------------------------------------------
    
    def concurrency_change(self, active: int, peak: int, target: int):
        """Log concurrency level change."""
        self.logger.debug(
            f"{self._elapsed()} CONCURRENCY | "
            f"active={active:3d} | peak={peak:3d} | target={target:3d}"
        )
    
    def queue_sample(self, depth: int, wait_time_ms: float):
        """Log queue depth sample."""
        self.logger.debug(
            f"{self._elapsed()} QUEUE | depth={depth:3d} | wait={wait_time_ms:7.2f}ms"
        )
    
    def rate_limit(self, target_rps: float, actual_rps: float, adjustment_ms: float):
        """Log rate limiting adjustment."""
        self.logger.debug(
            f"{self._elapsed()} RATE LIMIT | "
            f"target={target_rps:.2f} rps | actual={actual_rps:.2f} rps | "
            f"adjust={adjustment_ms:.2f}ms"
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
    ):
        """Log step start in stepping/ramp-up modes."""
        self.logger.info(
            f"{self._elapsed()} STEP {step}/{total_steps} START | "
            f"concurrency={concurrency} | requests={requests} | duration={duration:.1f}s"
        )
    
    def step_complete(self, step: int, total_steps: int, completed: int, elapsed: float):
        """Log step completion."""
        self.logger.info(
            f"{self._elapsed()} STEP {step}/{total_steps} DONE  | "
            f"completed={completed} | elapsed={elapsed:.2f}s"
        )
    
    def stage_start(
        self, 
        stage: int, 
        total_stages: int, 
        start_threads: int, 
        end_threads: int, 
        duration: float
    ):
        """Log ultimate thread group stage start."""
        if end_threads > start_threads:
            direction = "↑"
        elif end_threads < start_threads:
            direction = "↓"
        else:
            direction = "→"
        self.logger.info(
            f"{self._elapsed()} STAGE {stage}/{total_stages} | "
            f"{start_threads} {direction} {end_threads} over {duration:.1f}s"
        )
    
    # -------------------------------------------------------------------------
    # Spike Mode Logging
    # -------------------------------------------------------------------------
    
    def spike_phase(self, phase: str, concurrency: int, duration: float = 0):
        """Log spike test phase."""
        self.logger.info(
            f"{self._elapsed()} SPIKE {phase.upper()} | "
            f"concurrency={concurrency} | duration={duration:.1f}s"
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
    ):
        """Log duration mode progress."""
        remaining = target - elapsed
        self.logger.debug(
            f"{self._elapsed()} DURATION | "
            f"elapsed={elapsed:.1f}s | remaining={remaining:.1f}s | "
            f"sent={requests_sent} | active={active}"
        )
    
    # -------------------------------------------------------------------------
    # Warmup Logging
    # -------------------------------------------------------------------------
    
    def warmup_start(self, count: int):
        """Log warmup phase start."""
        self.logger.info(f"{self._elapsed()} WARMUP START | count={count}")
    
    def warmup_request(self, request_num: int, latency_ms: float, success: bool):
        """Log warmup request."""
        status = "OK" if success else "FAIL"
        self.logger.debug(
            f"{self._elapsed()} WARMUP #{request_num} | "
            f"latency={latency_ms:.2f}ms | {status}"
        )
    
    def warmup_complete(self, count: int, total_time_ms: float):
        """Log warmup completion."""
        self.logger.info(
            f"{self._elapsed()} WARMUP DONE | "
            f"count={count} | total={total_time_ms:.2f}ms"
        )
    
    # -------------------------------------------------------------------------
    # Summary Logging
    # -------------------------------------------------------------------------
    
    def summary(self, title: str, metrics: dict):
        """Log summary with metrics."""
        self.logger.info(f"\n{'-'*50}")
        self.logger.info(f"{title}")
        self.logger.info(f"{'-'*50}")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")


# =============================================================================
# Global Logger Management
# =============================================================================

_debug_logger: Optional[DebugLogger] = None


def get_debug_logger(console_output: bool = False) -> DebugLogger:
    """Get or create the global debug logger."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(console_output=console_output)
    return _debug_logger


def reset_debug_logger(console_output: bool = False) -> DebugLogger:
    """Reset the debug logger with new settings."""
    global _debug_logger
    _debug_logger = DebugLogger(console_output=console_output)
    return _debug_logger


def enable_debug_logging(level: int = logging.DEBUG, console_output: bool = False):
    """Enable debug logging at specified level."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(console_output=console_output)
    _debug_logger.set_level(level)
    if console_output and not _debug_logger.console_output:
        _debug_logger.enable_console()


def disable_debug_logging():
    """Disable debug logging."""
    logger = get_debug_logger()
    logger.set_level(logging.WARNING)
