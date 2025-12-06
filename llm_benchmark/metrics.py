"""
Metrics collection and calculation for LLM benchmarks.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    latency: float  # seconds
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    response: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""
    # Configuration
    model_name: str
    model_type: str
    scenario_name: str
    total_requests: int
    concurrency: int
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: float = 0.0  # seconds
    
    # Request metrics
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Latency metrics (seconds)
    latencies: List[float] = field(default_factory=list)
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0
    
    # Throughput metrics
    requests_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    
    # Token metrics
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_tokens_per_request: float = 0.0
    
    # Derived metrics
    ns_per_token: float = 0.0  # nanoseconds per token
    
    # Raw data
    request_metrics: List[RequestMetrics] = field(default_factory=list)
    responses: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate(self) -> None:
        """Calculate aggregated metrics from request metrics."""
        if not self.request_metrics:
            return
        
        # Filter successful requests
        successful = [m for m in self.request_metrics if m.success]
        self.successful_requests = len(successful)
        self.failed_requests = len(self.request_metrics) - self.successful_requests
        
        if not successful:
            return
        
        # Latency metrics
        self.latencies = [m.latency for m in successful]
        self.avg_latency = statistics.mean(self.latencies)
        self.p50_latency = float(np.percentile(self.latencies, 50))
        self.p90_latency = float(np.percentile(self.latencies, 90))
        self.p95_latency = float(np.percentile(self.latencies, 95))
        self.p99_latency = float(np.percentile(self.latencies, 99))
        self.min_latency = min(self.latencies)
        self.max_latency = max(self.latencies)
        
        # Token metrics
        self.total_tokens = sum(m.tokens for m in successful)
        self.total_prompt_tokens = sum(m.prompt_tokens for m in successful)
        self.total_completion_tokens = sum(m.completion_tokens for m in successful)
        
        if self.successful_requests > 0:
            self.avg_tokens_per_request = self.total_tokens / self.successful_requests
        
        # Throughput metrics
        if self.duration > 0:
            self.requests_per_sec = self.successful_requests / self.duration
            if self.total_tokens > 0:
                self.tokens_per_sec = self.total_tokens / self.duration
                # nanoseconds per token
                self.ns_per_token = (self.duration * 1e9) / self.total_tokens
        
        # Collect responses
        self.responses = [m.response for m in successful if m.response is not None]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "scenario_name": self.scenario_name,
            "timestamp": self.start_time.isoformat() if self.start_time else None,
            "configuration": {
                "total_requests": self.total_requests,
                "concurrency": self.concurrency
            },
            "results": {
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "duration_seconds": round(self.duration, 3),
                "requests_per_sec": round(self.requests_per_sec, 2),
                "tokens_per_sec": round(self.tokens_per_sec, 2) if self.tokens_per_sec else None,
                "ns_per_token": round(self.ns_per_token, 2) if self.ns_per_token else None
            },
            "latency": {
                "avg_seconds": round(self.avg_latency, 4),
                "p50_seconds": round(self.p50_latency, 4),
                "p95_seconds": round(self.p95_latency, 4),
                "p99_seconds": round(self.p99_latency, 4),
                "min_seconds": round(self.min_latency, 4),
                "max_seconds": round(self.max_latency, 4)
            },
            "tokens": {
                "total": self.total_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "avg_per_request": round(self.avg_tokens_per_request, 2)
            }
        }


class MetricsCollector:
    """Collects and manages benchmark metrics."""
    
    def __init__(self):
        self.benchmarks: List[BenchmarkMetrics] = []
    
    def create_benchmark(
        self,
        model_name: str,
        model_type: str,
        scenario_name: str,
        total_requests: int,
        concurrency: int
    ) -> BenchmarkMetrics:
        """Create a new benchmark metrics instance."""
        benchmark = BenchmarkMetrics(
            model_name=model_name,
            model_type=model_type,
            scenario_name=scenario_name,
            total_requests=total_requests,
            concurrency=concurrency,
            start_time=datetime.now()
        )
        self.benchmarks.append(benchmark)
        return benchmark
    
    def add_request_metric(
        self,
        benchmark: BenchmarkMetrics,
        latency: float,
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        response: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a request metric to a benchmark."""
        metric = RequestMetrics(
            latency=latency,
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            error=error,
            response=response
        )
        benchmark.request_metrics.append(metric)
    
    def finalize_benchmark(self, benchmark: BenchmarkMetrics, duration: float) -> None:
        """Finalize a benchmark and calculate metrics."""
        benchmark.end_time = datetime.now()
        benchmark.duration = duration
        benchmark.calculate()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all benchmarks."""
        return {
            "total_benchmarks": len(self.benchmarks),
            "benchmarks": [b.to_dict() for b in self.benchmarks]
        }
