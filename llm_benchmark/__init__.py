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
    ScenarioConfig,
    BenchmarkConfig,
    load_config,
)
from .engine import BenchmarkEngine, run_benchmark
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
