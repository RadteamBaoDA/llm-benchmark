"""
Core benchmark engine for running load tests against LLM APIs.
"""

import asyncio
import json
import os
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


class BenchmarkEngine:
    """Core engine for running LLM benchmarks."""
    
    def __init__(self, config: BenchmarkConfig, enable_timeseries: bool = True):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.client: Optional[httpx.AsyncClient] = None
        self.enable_timeseries = enable_timeseries
        self.timeseries_writer: Optional[TimeseriesWriter] = None
        self.timeseries_files: List[str] = []
        self._active_requests: int = 0
        
        if enable_timeseries:
            self.timeseries_writer = TimeseriesWriter(
                output_dir=config.output_dir,
                format="csv"
            )
    
    async def _make_request(
        self,
        mock_request: MockRequest,
        capture_response: bool = False,
        timeout: Optional[int] = None
    ) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int], Optional[Dict[str, Any]], Optional[str]]:
        """
        Make a single API request and return metrics.
        
        Returns:
            Tuple of (latency, total_tokens, prompt_tokens, completion_tokens, response, error)
        """
        url = f"{self.config.api.base_url.rstrip('/')}{mock_request.endpoint}"
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.config.api.api_key:
            headers["Authorization"] = f"Bearer {self.config.api.api_key}"
        
        # Use provided timeout or fall back to config
        request_timeout = timeout if timeout is not None else self.config.api.timeout
        
        t0 = time.perf_counter()
        
        try:
            response = await self.client.post(
                url,
                headers=headers,
                json=mock_request.payload,
                timeout=request_timeout
            )
            latency = time.perf_counter() - t0
            response.raise_for_status()
            
            data = response.json()
            
            # Extract token counts based on model type
            tokens, prompt_tokens, completion_tokens = self._extract_tokens(data)
            
            return latency, tokens, prompt_tokens, completion_tokens, data if capture_response else None, None
            
        except httpx.HTTPStatusError as e:
            latency = time.perf_counter() - t0
            return latency, None, None, None, None, f"HTTP {e.response.status_code}: {str(e)}"
        except Exception as e:
            latency = time.perf_counter() - t0
            return latency, None, None, None, None, str(e)
    
    def _extract_tokens(self, data: Dict[str, Any]) -> Tuple[int, int, int]:
        """Extract token counts from API response."""
        usage = data.get("usage", {})
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get(
            "total_tokens",
            prompt_tokens + completion_tokens
        )
        
        return total_tokens, prompt_tokens, completion_tokens
    
    async def run_scenario(
        self,
        scenario: ScenarioConfig,
        quiet: bool = False
    ) -> BenchmarkMetrics:
        """Run a single benchmark scenario."""
        # Check if scenario is enabled
        if not scenario.enabled:
            print(f"\n⏭️  Skipping disabled scenario: {scenario.name}")
            return None
        
        print(f"\n🚀 Running scenario: {scenario.name}")
        if scenario.description:
            print(f"   Description: {scenario.description}")
        print(f"   Requests: {scenario.requests}, Concurrency: {scenario.concurrency}")
        print(f"   Warmup: {scenario.warmup_requests}, Timeout: {scenario.timeout}s")
        
        # Start timeseries recording
        timeseries_file = None
        if self.timeseries_writer:
            timeseries_file = self.timeseries_writer.start_scenario(
                scenario.name, 
                self.config.model.name
            )
            print(f"   📊 Timeseries: {timeseries_file}")
            self.timeseries_files.append(timeseries_file)
        
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
        
        # Semaphore for concurrency control
        sem = asyncio.Semaphore(scenario.concurrency)
        self._active_requests = 0
        
        # Use scenario-specific timeout
        request_timeout = scenario.timeout
        
        async def worker(mock_request: MockRequest):
            async with sem:
                self._active_requests += 1
                current_concurrent = self._active_requests
                
                result = await self._make_request(
                    mock_request,
                    capture_response=self.config.capture_responses,
                    timeout=request_timeout
                )
                latency, tokens, prompt_tokens, completion_tokens, response, error = result
                
                self._active_requests -= 1
                
                # Record to timeseries
                if self.timeseries_writer:
                    self.timeseries_writer.record(
                        scenario_name=scenario.name,
                        model_name=self.config.model.name,
                        model_type=self.config.model.type,
                        latency=latency if latency else 0,
                        success=error is None,
                        tokens=tokens if tokens else 0,
                        prompt_tokens=prompt_tokens if prompt_tokens else 0,
                        completion_tokens=completion_tokens if completion_tokens else 0,
                        error=error,
                        concurrent_requests=current_concurrent
                    )
                
                self.metrics_collector.add_request_metric(
                    benchmark=benchmark,
                    latency=latency if latency else 0,
                    tokens=tokens if tokens else 0,
                    prompt_tokens=prompt_tokens if prompt_tokens else 0,
                    completion_tokens=completion_tokens if completion_tokens else 0,
                    success=error is None,
                    error=error,
                    response=response
                )
        
        # Run benchmark
        async with httpx.AsyncClient(http2=True, timeout=None) as self.client:
            # Warm-up requests
            if mock_requests and scenario.warmup_requests > 0:
                print(f"   🔥 Running {scenario.warmup_requests} warmup request(s)...")
                warmup_count = min(scenario.warmup_requests, len(mock_requests))
                for i in range(warmup_count):
                    await self._make_request(mock_requests[i % len(mock_requests)], capture_response=False)
            
            # Start timing
            start_time = time.perf_counter()
            
            # Create tasks
            tasks = [asyncio.create_task(worker(req)) for req in mock_requests]
            
            # Run with progress bar if available and not quiet
            if TQDM_AVAILABLE and not quiet:
                await tqdm_asyncio.gather(*tasks, desc=f"  {scenario.name}")
            else:
                await asyncio.gather(*tasks)
            
            # End timing
            end_time = time.perf_counter()
            duration = end_time - start_time
        
        # Finalize metrics
        self.metrics_collector.finalize_benchmark(benchmark, duration)
        
        # End timeseries recording
        if self.timeseries_writer:
            self.timeseries_writer.end_scenario()
        
        # Print results
        self._print_results(benchmark)
        
        return benchmark
    
    async def run_all_scenarios(self, quiet: bool = False) -> List[BenchmarkMetrics]:
        """Run all configured scenarios."""
        results = []
        
        if not self.config.scenarios:
            # Run with default settings
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
            # Filter enabled scenarios
            enabled_scenarios = [s for s in self.config.scenarios if s.enabled]
            disabled_count = len(self.config.scenarios) - len(enabled_scenarios)
            
            if disabled_count > 0:
                print(f"\n📋 {len(enabled_scenarios)} scenarios enabled, {disabled_count} disabled")
            
            # Run each enabled scenario
            for scenario in self.config.scenarios:
                result = await self.run_scenario(scenario, quiet)
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
    
    def _print_results(self, metrics: BenchmarkMetrics) -> None:
        """Print benchmark results to console."""
        print(f"\n{'='*60}")
        print(f"📊 Results for: {metrics.scenario_name}")
        print(f"{'='*60}")
        
        # Success/failure
        print(f"\n✅ {metrics.successful_requests}/{metrics.total_requests} requests succeeded in {metrics.duration:.2f}s")
        
        if metrics.failed_requests > 0:
            print(f"❌ {metrics.failed_requests} requests failed")
        
        if metrics.successful_requests == 0:
            print("⚠️  No successful requests to analyze")
            return
        
        # Throughput
        print(f"\n📈 Throughput:")
        print(f"   Requests/s:     {metrics.requests_per_sec:>10.2f}")
        if metrics.tokens_per_sec:
            print(f"   Tokens/s:       {metrics.tokens_per_sec:>10.2f}")
            print(f"   ns/Token:       {metrics.ns_per_token:>10.2f}")
        
        # Latency
        print(f"\n⏱️  Latency:")
        print(f"   Average:        {metrics.avg_latency:>10.4f}s")
        print(f"   P50 (median):   {metrics.p50_latency:>10.4f}s")
        print(f"   P95:            {metrics.p95_latency:>10.4f}s")
        print(f"   P99:            {metrics.p99_latency:>10.4f}s")
        print(f"   Min:            {metrics.min_latency:>10.4f}s")
        print(f"   Max:            {metrics.max_latency:>10.4f}s")
        
        # Tokens
        if metrics.total_tokens > 0:
            print(f"\n🔤 Tokens:")
            print(f"   Total:          {metrics.total_tokens:>10}")
            print(f"   Prompt:         {metrics.total_prompt_tokens:>10}")
            print(f"   Completion:     {metrics.total_completion_tokens:>10}")
            print(f"   Avg/Request:    {metrics.avg_tokens_per_request:>10.2f}")
        
        print()
    
    def save_responses(self, metrics: BenchmarkMetrics, filename: Optional[str] = None) -> str:
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


async def run_benchmark(config: BenchmarkConfig, scenarios_only: bool = False, enable_timeseries: bool = True) -> List[BenchmarkMetrics]:
    """
    Main entry point for running benchmarks.
    
    Args:
        config: Benchmark configuration
        scenarios_only: If True, only run configured scenarios. If False, run default if no scenarios.
        enable_timeseries: If True, record timeseries metrics to file
    
    Returns:
        List of benchmark metrics for each scenario
    """
    engine = BenchmarkEngine(config, enable_timeseries=enable_timeseries)
    
    print("\n" + "="*60)
    print("🔥 LLM Benchmark Tool")
    print("="*60)
    print(f"\n📝 Configuration:")
    print(f"   Base URL:       {config.api.base_url}")
    print(f"   Model:          {config.model.name}")
    print(f"   Model Type:     {config.model.type}")
    print(f"   Scenarios:      {len(config.scenarios) if config.scenarios else 'default'}")
    
    results = await engine.run_all_scenarios(quiet=config.quiet)
    
    # Save responses if capturing is enabled
    if config.capture_responses:
        for result in results:
            if result.responses:
                engine.save_responses(result)
    
    return results
