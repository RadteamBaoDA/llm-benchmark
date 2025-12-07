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

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .config import BenchmarkConfig, ScenarioConfig
from .event_bus import EventBus
from .events import (
    RequestCompleted,
    RequestEvent,
    RequestFailed,
    RequestStarted,
    RequestSubmitted,
    ScenarioCompleted,
    ScenarioStarted,
)
from .http_logging import HttpLogger
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
from .request_executor import RequestExecutor


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
        
        # HTTP request/response logging
        self.http_logger = HttpLogger(config.logging)
        
        # Event bus for lifecycle notifications
        self.event_bus = EventBus()

        # Request executor
        self.request_executor = RequestExecutor(config, self.event_bus, self.http_logger)
        
        # Concurrent request tracking
        self._active_requests: int = 0
        self._active_lock = asyncio.Lock()
        self._peak_concurrent: int = 0
        
        # Rich Console
        self.console = Console()
        
        # Queue metrics
        self.queue_metrics: Optional[QueueMetrics] = None
        
        # Request timing tracking
        self._request_submit_times: Dict[int, float] = {}
        self._request_start_times: Dict[int, float] = {}
        self._benchmark_start_time: float = 0.0
        
        # Request/Response capture for HTML report
        self._capture_data: List[Dict[str, Any]] = []
        self._capture_file: Optional[str] = None
        
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

        # Publish scenario start
        await self.event_bus.publish(ScenarioStarted(scenario=scenario.name))
        
        # Debug logging
        if self.debug:
            profile = self._get_load_profile(scenario)
            self._log_mode_start(scenario, mode, profile)
        
        # Setup timeseries
        timeseries_file = self._setup_timeseries(scenario)
        
        # Setup request/response capture
        self._setup_capture(scenario)
        
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
        task_id = None
        if pbar:
            task_id = pbar.add_task(
                f"  {scenario.name}", 
                total=len(mock_requests), 
                active=0, 
                errors=0
            )
            pbar.start()
        
        # Create worker function
        worker = self._create_worker(scenario, benchmark, pbar, task_id)
        
        # Execute benchmark
        async with self._create_http_client(scenario.concurrency) as self.client:
            await self._run_warmup(mock_requests, scenario)
            
            self._benchmark_start_time = time.perf_counter()
            
            if self.debug:
                self._debug_logger.mode_phase("EXECUTION", f"Starting {mode.value} mode")
            
            try:
                await self._execute_mode(mode, mock_requests, scenario, worker)
            finally:
                if pbar:
                    pbar.stop()
            
            duration = time.perf_counter() - self._benchmark_start_time
        
        # Finalize
        self._finalize_scenario(benchmark, duration, scenario)

        await self.event_bus.publish(
            ScenarioCompleted(
                scenario=scenario.name,
                duration=duration,
                success=benchmark.failed_requests == 0,
                error=None if benchmark.failed_requests == 0 else "Failed requests present",
            )
        )
        
        return benchmark
    
    def _print_scenario_header(self, scenario: ScenarioConfig, mode: BenchmarkMode):
        """Print scenario header information using Rich Panel."""
        info = [
            f"[bold]Description:[/bold] {scenario.description}" if scenario.description else "",
            f"[bold]Requests:[/bold] {scenario.requests}, [bold]Concurrency:[/bold] {scenario.concurrency}",
            f"[bold]Warmup:[/bold] {scenario.warmup_requests}, [bold]Timeout:[/bold] {scenario.timeout}s",
            f"[bold]Mode:[/bold] {mode.value}"
        ]
        if self.debug:
            info.append("[bold yellow]🐛 Debug logging: ENABLED[/bold yellow]")
            
        content = "\n".join(filter(None, info))
        self.console.print(Panel(
            content,
            title=f"🚀 Running scenario: [bold cyan]{scenario.name}[/bold cyan]",
            border_style="blue",
            expand=False
        ))
    
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
    
    def _setup_capture(self, scenario: ScenarioConfig) -> None:
        """Setup request/response capture for the scenario."""
        if self.config.capture_request_response:
            self.start_capture(scenario.name, self.config.model.name)
            print(f"   💬 Capture: Enabled (request/response logging)")
    
    def _create_progress_bar(self, mock_requests: List[MockRequest], name: str, quiet: bool):
        """Create a progress bar if not quiet."""
        if not quiet:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TextColumn("[cyan]Active: {task.fields[active]}"),
                TextColumn("[red]Errors: {task.fields[errors]}"),
                refresh_per_second=10,
                console=self.console
            )
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
        pbar,
        task_id=None
    ):
        """Create worker function for request execution."""
        engine = self
        request_timeout = scenario.timeout
        # Streaming is supported for chat and vision model types (both use chat completions API)
        use_streaming = self.config.api.streaming and self.config.model.type in ("chat", "vision")
        
        async def worker(
            request_id: int, 
            mock_request: MockRequest, 
            sem: Optional[asyncio.Semaphore] = None
        ):
            """Execute a single request with proper concurrency tracking."""
            submit_time = time.perf_counter()
            engine._request_submit_times[request_id] = submit_time

            # Notify subscribers that a request was submitted
            await engine.event_bus.publish(
                RequestSubmitted(
                    request_id=request_id,
                    endpoint=mock_request.endpoint,
                    payload=mock_request.payload,
                )
            )
            
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
                
                # Update progress bar with active count
                if pbar and task_id is not None:
                    pbar.update(task_id, active=current_concurrent)

                # Use streaming or non-streaming based on config
                ttft_ms = None
                tpot_ms = None
                itl_ms = None
                streaming = False
                
                # Capture response if capture_request_response is enabled
                should_capture = engine.config.capture_request_response
                
                if use_streaming:
                    result = await engine.request_executor.make_streaming_request(
                        client=engine.client,
                        mock_request=mock_request,
                        capture_response=should_capture,
                        timeout=request_timeout,
                        request_id=request_id,
                        queue_metrics=engine.queue_metrics,
                        request_submit_times=engine._request_submit_times,
                        request_start_times=engine._request_start_times,
                    )
                    latency, tokens, prompt_tokens, completion_tokens, response, error, http_status, ttft_ms, tpot_ms, itl_ms = result
                    streaming = True
                else:
                    result = await engine.request_executor.make_request(
                        client=engine.client,
                        mock_request=mock_request,
                        capture_response=should_capture,
                        timeout=request_timeout,
                        request_id=request_id,
                        queue_metrics=engine.queue_metrics,
                        request_submit_times=engine._request_submit_times,
                        request_start_times=engine._request_start_times,
                    )
                    latency, tokens, prompt_tokens, completion_tokens, response, error, http_status = result
                    # For non-streaming requests:
                    # - TTFT is not measurable (all tokens arrive at once) - keep as None
                    # - TPOT can be estimated as total_latency / completion_tokens
                    if error is None and latency and completion_tokens and completion_tokens > 0:
                        tpot_ms = (latency * 1000) / completion_tokens  # Estimated TPOT
                
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
                        ttft_ms=ttft_ms,
                        tpot_ms=tpot_ms,
                        itl_ms=itl_ms,
                        streaming=streaming
                    )
                
                # Capture request/response for HTML report viewing
                if engine.config.capture_request_response:
                    engine.capture_request_response(
                        request_id=request_id,
                        prompt=mock_request.payload,
                        response_body=response,
                        latency_ms=(latency or 0) * 1000,
                        success=error is None,
                        error=error,
                        tokens=tokens or 0,
                        model_type=engine.config.model.type
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
                
                if pbar and task_id is not None:
                    pbar.update(
                        task_id, 
                        advance=1, 
                        active=remaining_concurrent,
                        errors=benchmark.failed_requests
                    )
                    
            except Exception as worker_error:
                # Log any exception in worker
                engine.http_logger.logger.error(
                    "REQ #%s WORKER ERROR: %s: %s",
                    request_id,
                    type(worker_error).__name__,
                    worker_error,
                )
                # Still record as failed request
                engine.metrics_collector.add_request_metric(
                    benchmark=benchmark,
                    latency=0,
                    tokens=0,
                    success=False,
                    error=f"Worker error: {worker_error}"
                )
                await engine._decrement_active()
                if pbar and task_id is not None:
                    pbar.update(task_id, advance=1, errors=benchmark.failed_requests)
                raise
                    

                    
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
            result = await self.request_executor.make_request(
                client=self.client,
                mock_request=mock_requests[i],
                capture_response=False,
                timeout=scenario.timeout,
                request_id=None,
                queue_metrics=self.queue_metrics,
                request_submit_times=self._request_submit_times,
                request_start_times=self._request_start_times,
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
        
        # Save captured request/response data
        if self.config.capture_request_response:
            self.save_capture()
        
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
        """Print benchmark results to console using Rich Table."""
        
        # Summary Panel
        summary = (
            f"✅ [green]{metrics.successful_requests}/{metrics.total_requests}[/green] requests succeeded\n"
            f"⏱️  Duration: [bold]{metrics.duration:.2f}s[/bold]"
        )
        if metrics.failed_requests > 0:
            summary += f"\n❌ [red]{metrics.failed_requests} requests failed[/red]"
            
        self.console.print(Panel(
            summary,
            title=f"📊 Results for: [bold]{metrics.scenario_name}[/bold]",
            border_style="green" if metrics.failed_requests == 0 else "yellow"
        ))
        
        if metrics.successful_requests == 0:
            self.console.print("[bold red]⚠️  No successful requests to analyze[/bold red]")
            return

        # Metrics Table
        table = Table(title="📈 Throughput & Latency", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Requests/sec", f"{metrics.requests_per_sec:.2f}")
        if metrics.avg_latency:
            table.add_row("Avg Latency", f"{metrics.avg_latency * 1000:.2f} ms")
        if metrics.p50_latency:
            table.add_row("P50 Latency", f"{metrics.p50_latency * 1000:.2f} ms")
        if metrics.p90_latency:
            table.add_row("P90 Latency", f"{metrics.p90_latency * 1000:.2f} ms")
        if metrics.p95_latency:
            table.add_row("P95 Latency", f"{metrics.p95_latency * 1000:.2f} ms")
        if metrics.p99_latency:
            table.add_row("P99 Latency", f"{metrics.p99_latency * 1000:.2f} ms")
        
        if metrics.total_tokens > 0:
            table.add_section()
            table.add_row("Total Tokens", f"{metrics.total_tokens:,}")
            table.add_row("Tokens/sec", f"{metrics.tokens_per_sec:.2f}")
            
        self.console.print(table)
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
    
    def start_capture(self, scenario_name: str, model_name: str) -> None:
        """Start capturing request/response data for a scenario."""
        if not self.config.capture_request_response:
            return
        
        self._capture_data = []
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_scenario = scenario_name.replace(" ", "_").replace("/", "-").replace(":", "-")
        safe_model = model_name.replace("/", "-").replace(":", "-")
        filename = f"capture_{safe_scenario}_{safe_model}_{timestamp}.json"
        
        self._capture_file = str(output_dir / filename)
    
    def capture_request_response(
        self,
        request_id: int,
        prompt: Any,
        response_body: Any,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
        tokens: int = 0,
        model_type: str = "chat"
    ) -> None:
        """Capture a request/response pair for later viewing in HTML report."""
        if not self.config.capture_request_response:
            return
        
        # Extract readable prompt based on model type
        if model_type == "chat":
            if isinstance(prompt, dict):
                messages = prompt.get("messages", [])
                prompt_text = "\n".join([
                    f"[{m.get('role', 'user')}]: {m.get('content', '')}"
                    for m in messages
                ])
            else:
                prompt_text = str(prompt)
        elif model_type == "embed":
            if isinstance(prompt, dict):
                prompt_text = prompt.get("input", str(prompt))
            else:
                prompt_text = str(prompt)
        elif model_type == "reranker":
            if isinstance(prompt, dict):
                query = prompt.get("query", "")
                docs = prompt.get("documents", [])
                prompt_text = f"Query: {query}\n\nDocuments:\n" + "\n".join([f"- {d}" for d in docs[:5]])
                if len(docs) > 5:
                    prompt_text += f"\n... and {len(docs) - 5} more"
            else:
                prompt_text = str(prompt)
        elif model_type == "vision":
            if isinstance(prompt, dict):
                messages = prompt.get("messages", [])
                prompt_parts = []
                for m in messages:
                    role = m.get('role', 'user')
                    content = m.get('content', '')
                    # Vision content can be a list with text and image_url
                    if isinstance(content, list):
                        text_parts = []
                        image_count = 0
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'text':
                                    text_parts.append(item.get('text', ''))
                                elif item.get('type') == 'image_url':
                                    image_count += 1
                        content_str = " ".join(text_parts)
                        if image_count > 0:
                            content_str += f" [+{image_count} image(s)]"
                        prompt_parts.append(f"[{role}]: {content_str}")
                    else:
                        prompt_parts.append(f"[{role}]: {content}")
                prompt_text = "\n".join(prompt_parts)
            else:
                prompt_text = str(prompt)
        else:
            prompt_text = str(prompt) if prompt else ""
        
        # Extract readable response
        if isinstance(response_body, dict):
            if model_type == "chat":
                # Handle streaming response with chunks
                if "chunks" in response_body:
                    chunks = response_body.get("chunks", [])
                    # Extract content from streaming chunks
                    content_parts = []
                    for chunk in chunks:
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                content_parts.append(content)
                    response_text = "".join(content_parts) if content_parts else "(No content in streaming response)"
                else:
                    # Regular non-streaming response
                    choices = response_body.get("choices", [])
                    if choices:
                        response_text = choices[0].get("message", {}).get("content", "")
                        if not response_text:
                            response_text = "(Empty response content)"
                    else:
                        response_text = json.dumps(response_body, indent=2)
            elif model_type == "embed":
                data = response_body.get("data", [])
                if data:
                    embedding = data[0].get("embedding", [])
                    if len(embedding) > 2:
                        response_text = f"Embedding vector (dim={len(embedding)}): [{embedding[0]:.6f}, {embedding[1]:.6f}, ... , {embedding[-1]:.6f}]"
                    elif embedding:
                        response_text = f"Embedding vector (dim={len(embedding)}): {embedding}"
                    else:
                        response_text = "(Empty embedding response)"
                else:
                    response_text = json.dumps(response_body, indent=2) if response_body else "(No embedding data)"
            elif model_type == "reranker":
                results = response_body.get("results", [])
                if results:
                    response_text = "Reranked Results:\n" + "\n".join([
                        f"  {i+1}. Score: {r.get('relevance_score', r.get('score', 0)):.4f}"
                        for i, r in enumerate(results[:5])
                    ])
                    if len(results) > 5:
                        response_text += f"\n  ... and {len(results) - 5} more results"
                else:
                    response_text = json.dumps(response_body, indent=2) if response_body else "(No reranker results)"
            elif model_type == "vision":
                # Vision uses same format as chat
                if "chunks" in response_body:
                    chunks = response_body.get("chunks", [])
                    content_parts = []
                    for chunk in chunks:
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                content_parts.append(content)
                    response_text = "".join(content_parts) if content_parts else "(No content in streaming response)"
                else:
                    choices = response_body.get("choices", [])
                    if choices:
                        response_text = choices[0].get("message", {}).get("content", "")
                        if not response_text:
                            response_text = "(Empty response content)"
                    else:
                        response_text = json.dumps(response_body, indent=2)
            else:
                response_text = json.dumps(response_body, indent=2)
        elif response_body:
            response_text = str(response_body)
        else:
            response_text = error or "No response"
        
        capture_record = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt_text,
            "response": response_text,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "error": error,
            "tokens": tokens,
            "model_type": model_type,
            "raw_request": prompt if isinstance(prompt, dict) else None,
            "raw_response": response_body if isinstance(response_body, dict) else None
        }
        
        self._capture_data.append(capture_record)
    
    def save_capture(self) -> Optional[str]:
        """Save captured request/response data to file."""
        if not self.config.capture_request_response or not self._capture_data:
            return None
        
        if not self._capture_file:
            return None
        
        with open(self._capture_file, 'w', encoding='utf-8') as f:
            json.dump(self._capture_data, f, indent=2, default=str)
        
        print(f"💬 Request/Response data saved to: {self._capture_file}")
        return self._capture_file


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
    
    return results
