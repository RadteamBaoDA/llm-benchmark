"""
LLM Benchmark Tool - A comprehensive benchmarking tool for OpenAI-compatible APIs.

Supports benchmarking for:
- Chat models
- Embedding models
- Reranker models
- Vision models
"""

__version__ = "1.0.0"
__author__ = "LLM Benchmark Tool"

from .config import (
    APIConfig,
    ModelConfig,
    MockDataConfig,
    LoggingConfig,
    ScenarioConfig,
    BenchmarkConfig,
    load_config,
)

# Import from modular structure
from .debug_logger import (
    DebugLogger,
    enable_debug_logging,
    disable_debug_logging,
    get_debug_logger,
    reset_debug_logger,
    get_structlog_logger,
    bind_contextvars,
    clear_contextvars,
    unbind_contextvars,
    configure_structlog,
)
from .modes import (
    BenchmarkMode,
    LoadProfile,
    QueueMetrics,
)
from .mode_runners import (
    ModeRunner,
    RampUpRunner,
    SteppingRunner,
    SpikeRunner,
    ConstantRateRunner,
    ArrivalsRunner,
    UltimateRunner,
    DurationRunner,
    get_mode_runner,
)
from .engine import (
    BenchmarkEngine, 
    run_benchmark,
)
from .metrics import RequestMetrics, BenchmarkMetrics, MetricsCollector
from .exporters import export_results
from .timeseries import TimeseriesWriter, TimeseriesReader, load_all_timeseries
from .html_report import HTMLReportGenerator, generate_html_report

__all__ = [
    # Config
    "APIConfig",
    "ModelConfig",
    "MockDataConfig",
    "ScenarioConfig",
    "BenchmarkConfig",
    "load_config",
    # Engine
    "BenchmarkEngine",
    "run_benchmark",
    # Modes
    "BenchmarkMode",
    "LoadProfile",
    "QueueMetrics",
    # Mode Runners
    "ModeRunner",
    "RampUpRunner",
    "SteppingRunner",
    "SpikeRunner",
    "ConstantRateRunner",
    "ArrivalsRunner",
    "UltimateRunner",
    "DurationRunner",
    "get_mode_runner",
    # Debug
    "DebugLogger",
    "enable_debug_logging",
    "disable_debug_logging",
    "get_debug_logger",
    "reset_debug_logger",
    "get_structlog_logger",
    "bind_contextvars",
    "clear_contextvars",
    "unbind_contextvars",
    "configure_structlog",
    # Metrics
    "RequestMetrics",
    "BenchmarkMetrics",
    "MetricsCollector",
    # Exporters
    "export_results",
    # Timeseries
    "TimeseriesWriter",
    "TimeseriesReader",
    "load_all_timeseries",
    # HTML Report
    "HTMLReportGenerator",
    "generate_html_report",
]
