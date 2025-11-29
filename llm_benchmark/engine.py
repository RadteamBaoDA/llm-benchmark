"""
Core benchmark engine for running load tests against LLM APIs.
Optimized for true parallel execution and queue behavior testing.

Based on JMeter Thread Group specifications for comprehensive load testing.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from tqdm.asyncio import tqdm_asyncio
    TQDM_AVAILABLE = True
except ImportError:
    tqdm_asyncio = None
    TQDM_AVAILABLE = False

from .config import BenchmarkConfig, ScenarioConfig
from .metrics import BenchmarkMetrics, MetricsCollector, RequestMetrics
from .mock_data import MockRequest, get_mock_generator
from .timeseries import TimeseriesWriter

# Import from refactored modules
from .debug_logger import (
    DebugLogger,
    get_debug_logger,
    reset_debug_logger,
    enable_debug_logging,
    disable_debug_logging,
)
from .modes import BenchmarkMode, LoadProfile, QueueMetrics
from .mode_runners import get_mode_runner


# Re-export for backward compatibility
__all__ = [
    'BenchmarkEngine',
    'run_benchmark',
    'BenchmarkMode',
    'LoadProfile',
    'QueueMetrics',
    'DebugLogger',
    'get_debug_logger',
    'reset_debug_logger',
    'enable_debug_logging',
    'disable_debug_logging',
]


class BenchmarkEngine:
    """
    Core engine for running LLM benchmarks.
    
    Optimized for true parallel execution to accurately test:
    - Inference server queue behavior (Ollama, vLLM, TGI)
    - Concurrent request handling
    - Queue depth and wait times
    """
    
    def __init__(
        self, 
        config: BenchmarkConfig, 
        enable_timeseries: bool = True, 
        debug: bool = False, 
        debug_console: bool = False
    ):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.client: Optional[httpx.AsyncClient] = None
        self.enable_timeseries = enable_timeseries
        self.timeseries_writer: Optional[TimeseriesWriter] = None
        self.timeseries_files: List[str] = []
        
        # Debug logging
        self.debug = debug
        self.debug_console = debug_console
        self._setup_debug_logging()
        
        # Concurrent request tracking
        self._active_requests: int = 0
        self._active_lock = asyncio.Lock()
        self._peak_concurrent: int = 0
        
        # Queue metrics
        self.queue_metrics: Optional[QueueMetrics] = None
        
        # Request timing tracking
        self._request_submit_times: Dict[int, float] = {}
        self._request_start_times: Dict[int, float] = {}
        self._benchmark_start_time: float = 0.0
        
        if enable_timeseries:
            self.timeseries_writer = TimeseriesWriter(
                output_dir=config.output_dir,
                format="csv"
            )
    
    def _setup_debug_logging(self):
        """Configure debug logging based on settings."""
        if self.debug:
            self._debug_logger = reset_debug_logger(console_output=self.debug_console)
            enable_debug_logging(logging.DEBUG, console_output=self.debug_console)
        else:
            self._debug_logger = get_debug_logger()
            disable_debug_logging()
    
    # =========================================================================
    # Concurrency Management
    # =========================================================================
    
    async def _increment_active(self) -> int:
        """Atomically increment active request count."""
        async with self._active_lock:
            self._active_requests += 1
            if self._active_requests > self._peak_concurrent:
                self._peak_concurrent = self._active_requests
            return self._active_requests
    
    async def _decrement_active(self) -> int:
        """Atomically decrement active request count."""
        async with self._active_lock:
            self._active_requests -= 1
            return self._active_requests
    
    # =========================================================================
    # HTTP Request Handling
    # =========================================================================
    
    async def _make_request(
        self,
        mock_request: MockRequest,
        capture_response: bool = False,
        timeout: Optional[int] = None,
        request_id: Optional[int] = None
    ) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int], 
               Optional[Dict[str, Any]], Optional[str], Optional[int]]:
        """
        Make a single API request and return metrics.
        
        Returns:
            Tuple of (latency, total_tokens, prompt_tokens, completion_tokens, 
                     response, error, http_status)
        """
        url = f"{self.config.api.base_url.rstrip('/')}{mock_request.endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if self.config.api.api_key:
            headers["Authorization"] = f"Bearer {self.config.api.api_key}"
        
        request_timeout = timeout if timeout is not None else self.config.api.timeout
        
        if request_id is not None and request_id in self._request_submit_times:
            self._request_start_times[request_id] = time.perf_counter()
        
        t0 = time.perf_counter()
        http_status = None
        
        try:
            response = await self.client.post(
                url,
                headers=headers,
                json=mock_request.payload,
                timeout=request_timeout
            )
            latency = time.perf_counter() - t0
            http_status = response.status_code
            response.raise_for_status()
            
            data = response.json()
            tokens, prompt_tokens, completion_tokens = self._extract_tokens(data)
            
            return (latency, tokens, prompt_tokens, completion_tokens, 
                    data if capture_response else None, None, http_status)
            
        except httpx.HTTPStatusError as e:
            latency = time.perf_counter() - t0
            http_status = e.response.status_code
            error_msg = f"HTTP {http_status}"
            
            if http_status == 429:
                error_msg = "HTTP 429: Rate Limited / Queue Full"
                if self.queue_metrics:
                    self.queue_metrics.record_rejection()
            elif http_status == 503:
                error_msg = "HTTP 503: Service Unavailable / Server Overloaded"
                if self.queue_metrics:
                    self.queue_metrics.record_rejection()
            
            return latency, None, None, None, None, error_msg, http_status
            
        except httpx.TimeoutException:
            latency = time.perf_counter() - t0
            if self.queue_metrics:
                self.queue_metrics.record_timeout()
            return latency, None, None, None, None, f"Timeout after {request_timeout}s", None
            
        except Exception as e:
            latency = time.perf_counter() - t0
            return latency, None, None, None, None, str(e), None
    
    def _extract_tokens(self, data: Dict[str, Any]) -> Tuple[int, int, int]:
        """Extract token counts from API response."""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        return total_tokens, prompt_tokens, completion_tokens
    
    # =========================================================================
    # Load Profile
    # =========================================================================
    
    def _get_load_profile(self, scenario: ScenarioConfig) -> LoadProfile:
        """Extract load profile from scenario configuration."""
        return LoadProfile.from_scenario(scenario)
    
    # =========================================================================
    # Scenario Execution
    # =========================================================================
    
    async def run_scenario(
        self,
        scenario: ScenarioConfig,
        quiet: bool = False,
        mode: BenchmarkMode = BenchmarkMode.PARALLEL
    ) -> BenchmarkMetrics:
        """Run a single benchmark scenario with true parallel execution."""
        if not scenario.enabled:
            print(f"\n⏭️  Skipping disabled scenario: {scenario.name}")
            return None
        
        self._print_scenario_header(scenario, mode)
        self._init_scenario_state()
        
        # Debug logging
        if self.debug:
            profile = self._get_load_profile(scenario)
            self._log_mode_start(scenario, mode, profile)
        
        # Setup timeseries
        timeseries_file = self._setup_timeseries(scenario)
        
        # Create benchmark metrics
        benchmark = self.metrics_collector.create_benchmark(
            model_name=self.config.model.name,
            model_type=self.config.model.type,
            scenario_name=scenario.name,
            total_requests=scenario.requests,
            concurrency=scenario.concurrency
        )
        
        # Generate mock requests
        mock_generator = get_mock_generator(self.config.model, self.config.mock_data)
        mock_requests = mock_generator.generate(scenario.requests)
        
        # Create progress bar
        pbar = self._create_progress_bar(mock_requests, scenario.name, quiet)
        
        # Create worker function
        worker = self._create_worker(scenario, benchmark, pbar)
        
        # Execute benchmark
        async with self._create_http_client(scenario.concurrency) as self.client:
            await self._run_warmup(mock_requests, scenario)
            
            self._benchmark_start_time = time.perf_counter()
            
            if self.debug:
                self._debug_logger.mode_phase("EXECUTION", f"Starting {mode.value} mode")
            
            await self._execute_mode(mode, mock_requests, scenario, worker)
            
            duration = time.perf_counter() - self._benchmark_start_time
        
        if pbar:
            pbar.close()
        
        # Finalize
        self._finalize_scenario(benchmark, duration, scenario)
        
        return benchmark
    
    def _print_scenario_header(self, scenario: ScenarioConfig, mode: BenchmarkMode):
        """Print scenario header information."""
        print(f"\n🚀 Running scenario: {scenario.name}")
        if scenario.description:
            print(f"   Description: {scenario.description}")
        print(f"   Requests: {scenario.requests}, Concurrency: {scenario.concurrency}")
        print(f"   Warmup: {scenario.warmup_requests}, Timeout: {scenario.timeout}s")
        print(f"   Mode: {mode.value}")
        if self.debug:
            print(f"   🐛 Debug logging: ENABLED")
    
    def _init_scenario_state(self):
        """Initialize state for a new scenario run."""
        self.queue_metrics = QueueMetrics()
        self._peak_concurrent = 0
        self._active_requests = 0
        self._request_submit_times.clear()
        self._request_start_times.clear()
    
    def _log_mode_start(self, scenario: ScenarioConfig, mode: BenchmarkMode, profile: LoadProfile):
        """Log mode start with configuration."""
        config_dict = {
            "requests": scenario.requests,
            "concurrency": scenario.concurrency,
            "warmup_requests": scenario.warmup_requests,
            "timeout": scenario.timeout,
            "ramp_up_time": profile.ramp_up_time,
            "ramp_up_steps": profile.ramp_up_steps,
            "hold_time": profile.hold_time,
            "target_rps": profile.target_rps,
            "spike_multiplier": profile.spike_multiplier,
            "spike_duration": profile.spike_duration,
            "arrival_rate": profile.arrival_rate,
            "duration_seconds": profile.duration_seconds,
            "stages_count": len(profile.stages),
        }
        self._debug_logger.mode_start(mode.value, scenario.name, config_dict)
    
    def _setup_timeseries(self, scenario: ScenarioConfig) -> Optional[str]:
        """Setup timeseries recording for the scenario."""
        if self.timeseries_writer:
            timeseries_file = self.timeseries_writer.start_scenario(
                scenario.name, 
                self.config.model.name
            )
            print(f"   📊 Timeseries: {timeseries_file}")
            self.timeseries_files.append(timeseries_file)
            return timeseries_file
        return None
    
    def _create_progress_bar(self, mock_requests: List[MockRequest], name: str, quiet: bool):
        """Create a progress bar if available and not quiet."""
        if TQDM_AVAILABLE and not quiet:
            from tqdm import tqdm
            return tqdm(total=len(mock_requests), desc=f"  {name}")
        return None
    
    def _create_http_client(self, concurrency: int):
        """Create configured HTTP client."""
        limits = httpx.Limits(
            max_keepalive_connections=min(concurrency * 2, 200),
            max_connections=min(concurrency * 2, 200),
            keepalive_expiry=30.0
        )
        return httpx.AsyncClient(http2=True, timeout=None, limits=limits)
    
    def _create_worker(
        self, 
        scenario: ScenarioConfig, 
        benchmark: BenchmarkMetrics,
        pbar
    ):
        """Create worker function for request execution."""
        engine = self
        request_timeout = scenario.timeout
        
        async def worker(
            request_id: int, 
            mock_request: MockRequest, 
            sem: Optional[asyncio.Semaphore] = None
        ):
            """Execute a single request with proper concurrency tracking."""
            submit_time = time.perf_counter()
            engine._request_submit_times[request_id] = submit_time
            
            if engine.debug:
                engine._debug_logger.request_submit(request_id, engine._active_requests)
            
            if sem:
                await sem.acquire()
            
            try:
                current_concurrent = await engine._increment_active()
                
                if engine.debug:
                    engine._debug_logger.concurrency_change(
                        current_concurrent, 
                        engine._peak_concurrent, 
                        scenario.concurrency
                    )
                
                elapsed = time.perf_counter() - engine._benchmark_start_time
                engine.queue_metrics.add_queue_sample(elapsed, current_concurrent)
                
                wait_time = time.perf_counter() - submit_time
                engine.queue_metrics.add_wait_time(wait_time)
                
                if engine.debug:
                    engine._debug_logger.request_start(
                        request_id, wait_time * 1000, current_concurrent
                    )
                    if wait_time > 0.01:
                        engine._debug_logger.queue_sample(current_concurrent, wait_time * 1000)
                
                result = await engine._make_request(
                    mock_request,
                    capture_response=engine.config.capture_responses,
                    timeout=request_timeout,
                    request_id=request_id
                )
                latency, tokens, prompt_tokens, completion_tokens, response, error, http_status = result
                
                remaining_concurrent = await engine._decrement_active()
                
                if engine.debug:
                    if error:
                        engine._debug_logger.request_error(
                            request_id, error, (latency or 0) * 1000
                        )
                    else:
                        engine._debug_logger.request_complete(
                            request_id, (latency or 0) * 1000,
                            tokens or 0, error is None, remaining_concurrent
                        )
                
                if latency:
                    engine.queue_metrics.add_processing_time(latency)
                
                if engine.timeseries_writer:
                    engine.timeseries_writer.record(
                        scenario_name=scenario.name,
                        model_name=engine.config.model.name,
                        model_type=engine.config.model.type,
                        latency=latency if latency else 0,
                        success=error is None,
                        status_code=http_status,
                        tokens=tokens if tokens else 0,
                        prompt_tokens=prompt_tokens if prompt_tokens else 0,
                        completion_tokens=completion_tokens if completion_tokens else 0,
                        concurrent_requests=remaining_concurrent,
                        queue_time=wait_time
                    )
                
                engine.metrics_collector.add_request_metric(
                    benchmark=benchmark,
                    latency=latency if latency else 0,
                    tokens=tokens if tokens else 0,
                    prompt_tokens=prompt_tokens if prompt_tokens else 0,
                    completion_tokens=completion_tokens if completion_tokens else 0,
                    success=error is None,
                    error=error,
                    response=response
                )
                
                if pbar:
                    pbar.update(1)
                    
            finally:
                if sem:
                    sem.release()
        
        return worker
    
    async def _run_warmup(self, mock_requests: List[MockRequest], scenario: ScenarioConfig):
        """Run warmup requests."""
        if not mock_requests or scenario.warmup_requests <= 0:
            return
        
        print(f"   🔥 Running {scenario.warmup_requests} warmup request(s)...")
        if self.debug:
            self._debug_logger.warmup_start(scenario.warmup_requests)
        
        warmup_count = min(scenario.warmup_requests, len(mock_requests))
        warmup_start = time.perf_counter()
        
        for i in range(warmup_count):
            result = await self._make_request(
                mock_requests[i],
                capture_response=False,
                timeout=scenario.timeout
            )
            latency = result[0]
            error = result[5]
            
            if self.debug:
                self._debug_logger.warmup_request(
                    i + 1,
                    (latency or 0) * 1000,
                    error is None
                )
        
        warmup_elapsed = time.perf_counter() - warmup_start
        if self.debug:
            self._debug_logger.warmup_complete(warmup_count, warmup_elapsed * 1000)
    
    async def _execute_mode(
        self, 
        mode: BenchmarkMode, 
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker
    ):
        """Execute the appropriate benchmark mode."""
        # Get mode runner if available
        runner = get_mode_runner(mode, self.debug, self._debug_logger)
        
        if runner:
            profile = self._get_load_profile(scenario)
            await runner.run(mock_requests, scenario, worker, profile)
        elif mode == BenchmarkMode.PARALLEL:
            await self._run_parallel(mock_requests, worker)
        elif mode == BenchmarkMode.CONTROLLED:
            await self._run_controlled(mock_requests, scenario, worker)
        elif mode == BenchmarkMode.QUEUE_TEST:
            await self._run_queue_test(mock_requests, worker)
        else:
            # Default to parallel
            await self._run_parallel(mock_requests, worker)
    
    async def _run_parallel(self, mock_requests: List[MockRequest], worker):
        """Run all requests in parallel without limit."""
        if self.debug:
            self._debug_logger.mode_phase("PARALLEL", f"launching {len(mock_requests)} tasks")
        
        tasks = [
            asyncio.create_task(worker(i, req))
            for i, req in enumerate(mock_requests)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_controlled(
        self, 
        mock_requests: List[MockRequest], 
        scenario: ScenarioConfig,
        worker
    ):
        """Run with semaphore-controlled concurrency."""
        if self.debug:
            self._debug_logger.mode_phase(
                "CONTROLLED", 
                f"concurrency={scenario.concurrency}"
            )
        
        sem = asyncio.Semaphore(scenario.concurrency)
        tasks = [
            asyncio.create_task(worker(i, req, sem))
            for i, req in enumerate(mock_requests)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_queue_test(self, mock_requests: List[MockRequest], worker):
        """Run queue behavior test."""
        if self.debug:
            self._debug_logger.mode_phase("QUEUE_TEST", f"burst {len(mock_requests)} requests")
        
        tasks = [
            asyncio.create_task(worker(i, req))
            for i, req in enumerate(mock_requests)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _finalize_scenario(
        self, 
        benchmark: BenchmarkMetrics, 
        duration: float,
        scenario: ScenarioConfig
    ):
        """Finalize scenario metrics and logging."""
        self.queue_metrics.calculate()
        self.metrics_collector.finalize_benchmark(benchmark, duration)
        
        if self.timeseries_writer:
            self.timeseries_writer.end_scenario()
        
        if self.debug:
            stats = {
                "total_requests": scenario.requests,
                "successful": benchmark.successful_requests,
                "failed": benchmark.failed_requests,
                "duration_sec": f"{duration:.3f}",
                "peak_concurrent": self._peak_concurrent,
                "avg_latency_ms": f"{benchmark.avg_latency * 1000:.2f}" if benchmark.avg_latency else "N/A",
                "p95_latency_ms": f"{benchmark.p95_latency * 1000:.2f}" if benchmark.p95_latency else "N/A",
                "requests_per_sec": f"{benchmark.requests_per_sec:.2f}" if benchmark.requests_per_sec else "N/A",
                "queue_rejections": self.queue_metrics.rejection_count,
                "queue_timeouts": self.queue_metrics.timeout_count,
            }
            self._debug_logger.mode_end(stats)
        
        self._print_results(benchmark)
        self._print_queue_metrics()
    
    # =========================================================================
    # Batch Execution
    # =========================================================================
    
    async def run_all_scenarios(self, quiet: bool = False) -> List[BenchmarkMetrics]:
        """Run all configured scenarios."""
        results = []
        
        if not self.config.scenarios:
            default_scenario = ScenarioConfig(
                name="default",
                requests=self.config.default_requests,
                concurrency=self.config.default_concurrency,
                description="Default benchmark scenario"
            )
            result = await self.run_scenario(default_scenario, quiet)
            if result:
                results.append(result)
        else:
            enabled_scenarios = [s for s in self.config.scenarios if s.enabled]
            disabled_count = len(self.config.scenarios) - len(enabled_scenarios)
            
            if disabled_count > 0:
                print(f"\n⏭️  {disabled_count} scenario(s) disabled")
            
            for scenario in self.config.scenarios:
                mode = BenchmarkMode.from_string(scenario.mode) if scenario.mode else BenchmarkMode.PARALLEL
                result = await self.run_scenario(scenario, quiet, mode)
                if result:
                    results.append(result)
        
        return results
    
    async def run_single(
        self,
        requests: int,
        concurrency: int,
        scenario_name: str = "custom",
        quiet: bool = False
    ) -> BenchmarkMetrics:
        """Run a single benchmark with specified parameters."""
        scenario = ScenarioConfig(
            name=scenario_name,
            requests=requests,
            concurrency=concurrency
        )
        return await self.run_scenario(scenario, quiet)
    
    # =========================================================================
    # Output Methods
    # =========================================================================
    
    def _print_results(self, metrics: BenchmarkMetrics) -> None:
        """Print benchmark results to console."""
        print(f"\n{'='*60}")
        print(f"📊 Results for: {metrics.scenario_name}")
        print(f"{'='*60}")
        
        print(f"\n✅ {metrics.successful_requests}/{metrics.total_requests} "
              f"requests succeeded in {metrics.duration:.2f}s")
        
        if metrics.failed_requests > 0:
            print(f"❌ {metrics.failed_requests} requests failed")
        
        if metrics.successful_requests == 0:
            print("⚠️  No successful requests to analyze")
            return
        
        print(f"\n📈 Throughput:")
        print(f"   Requests/s:     {metrics.requests_per_sec:>10.2f}")
        if metrics.tokens_per_sec:
            print(f"   Tokens/s:       {metrics.tokens_per_sec:>10.2f}")
            print(f"   ns/Token:       {metrics.ns_per_token:>10.2f}")
        
        print(f"\n⏱️  Latency:")
        print(f"   Average:        {metrics.avg_latency:>10.4f}s")
        print(f"   P50 (median):   {metrics.p50_latency:>10.4f}s")
        print(f"   P95:            {metrics.p95_latency:>10.4f}s")
        print(f"   P99:            {metrics.p99_latency:>10.4f}s")
        print(f"   Min:            {metrics.min_latency:>10.4f}s")
        print(f"   Max:            {metrics.max_latency:>10.4f}s")
        
        if metrics.total_tokens > 0:
            print(f"\n🔤 Tokens:")
            print(f"   Total:          {metrics.total_tokens:>10}")
            print(f"   Prompt:         {metrics.total_prompt_tokens:>10}")
            print(f"   Completion:     {metrics.total_completion_tokens:>10}")
            print(f"   Avg/Request:    {metrics.avg_tokens_per_request:>10.2f}")
        
        print()
    
    def _print_queue_metrics(self) -> None:
        """Print queue behavior metrics."""
        if not self.queue_metrics:
            return
        
        qm = self.queue_metrics
        
        print(f"📋 Queue Metrics:")
        print(f"   Peak Concurrent:     {self._peak_concurrent:>10}")
        print(f"   Max Queue Depth:     {qm.max_observed_queue_depth:>10}")
        print(f"   Avg Queue Depth:     {qm.avg_queue_depth:>10.2f}")
        
        if qm.wait_times:
            try:
                import numpy as np
                avg_wait = np.mean(qm.wait_times) * 1000
                max_wait = np.max(qm.wait_times) * 1000
                print(f"   Avg Wait Time:       {avg_wait:>10.2f}ms")
                print(f"   Max Wait Time:       {max_wait:>10.2f}ms")
            except ImportError:
                avg_wait = sum(qm.wait_times) / len(qm.wait_times) * 1000
                max_wait = max(qm.wait_times) * 1000
                print(f"   Avg Wait Time:       {avg_wait:>10.2f}ms")
                print(f"   Max Wait Time:       {max_wait:>10.2f}ms")
        
        if qm.rejection_count > 0:
            print(f"   ⚠️  Rejections (429/503): {qm.rejection_count}")
        
        if qm.timeout_count > 0:
            print(f"   ⚠️  Timeouts:         {qm.timeout_count}")
        
        print()
    
    def save_responses(
        self, 
        metrics: BenchmarkMetrics, 
        filename: Optional[str] = None
    ) -> str:
        """Save captured responses to a JSON file."""
        if not metrics.responses:
            print("⚠️  No responses to save")
            return ""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"responses_{metrics.scenario_name}_{timestamp}.json"
        
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics.responses, f, indent=2)
        
        print(f"💾 Responses saved to: {filepath}")
        return str(filepath)


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_benchmark(
    config: BenchmarkConfig, 
    scenarios_only: bool = False, 
    enable_timeseries: bool = True,
    debug: bool = False,
    debug_console: bool = False
) -> List[BenchmarkMetrics]:
    """
    Main entry point for running benchmarks.
    
    Args:
        config: Benchmark configuration
        scenarios_only: If True, only run configured scenarios
        enable_timeseries: If True, record timeseries metrics to file
        debug: If True, enable detailed debug logging
        debug_console: If True, also output debug logs to console
    
    Returns:
        List of benchmark metrics for each scenario
    """
    engine = BenchmarkEngine(
        config, 
        enable_timeseries=enable_timeseries,
        debug=debug, 
        debug_console=debug_console
    )
    
    print("\n" + "="*60)
    print("🔥 LLM Benchmark Tool")
    print("="*60)
    print(f"\n📝 Configuration:")
    print(f"   Base URL:       {config.api.base_url}")
    print(f"   Model:          {config.model.name}")
    print(f"   Model Type:     {config.model.type}")
    print(f"   Scenarios:      {len(config.scenarios) if config.scenarios else 'default'}")
    if debug:
        print(f"   🐛 Debug Mode:   ENABLED (file: debug.log)")
        if debug_console:
            print(f"   🐛 Console:      ENABLED")
    
    results = await engine.run_all_scenarios(quiet=config.quiet)
    
    if config.capture_responses:
        for result in results:
            if result and result.responses:
                engine.save_responses(result)
    
    return results
