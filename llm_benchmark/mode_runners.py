"""
Execution mode runners for different load testing patterns.

Each runner implements a specific JMeter-style execution pattern.
"""

import asyncio
import random
import time
from typing import Callable, List, Optional

from .modes import LoadProfile, BenchmarkMode
from .debug_logger import DebugLogger
from .mock_data import MockRequest
from .config import ScenarioConfig


class ModeRunner:
    """Base class for execution mode runners."""
    
    def __init__(
        self, 
        debug: bool = False, 
        debug_logger: Optional[DebugLogger] = None
    ):
        self.debug = debug
        self._debug_logger = debug_logger
    
    async def run(
        self,
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker: Callable,
        profile: LoadProfile
    ):
        """Execute the mode. Override in subclasses."""
        raise NotImplementedError


class RampUpRunner(ModeRunner):
    """
    Execute requests with gradual ramp-up (JMeter ramp-up period style).
    
    Linearly increases concurrency from 0 to target over ramp_up_time.
    """
    
    async def run(
        self,
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker: Callable,
        profile: LoadProfile
    ):
        total = len(mock_requests)
        target_concurrency = scenario.concurrency
        ramp_time = profile.ramp_up_time if profile.ramp_up_time > 0 else 5.0
        ramp_steps = profile.ramp_up_steps if profile.ramp_up_steps > 1 else min(5, target_concurrency)
        
        print(f"   📈 Ramping up to {target_concurrency} threads over {ramp_time}s in {ramp_steps} steps...")
        
        if self.debug and self._debug_logger:
            self._debug_logger.mode_phase(
                "RAMP_UP", 
                f"target={target_concurrency}, time={ramp_time}s, steps={ramp_steps}"
            )
        
        step_duration = ramp_time / ramp_steps
        requests_per_step = total // ramp_steps
        
        all_tasks = []
        for step in range(ramp_steps):
            step_start = time.perf_counter()
            step_concurrency = max(1, (step + 1) * target_concurrency // ramp_steps)
            start_idx = step * requests_per_step
            end_idx = start_idx + requests_per_step if step < ramp_steps - 1 else total
            step_requests = end_idx - start_idx
            
            if self.debug and self._debug_logger:
                self._debug_logger.step_start(
                    step + 1, ramp_steps, step_concurrency, 
                    step_requests, step_duration
                )
            
            sem = asyncio.Semaphore(step_concurrency)
            step_tasks = [
                asyncio.create_task(worker(i, mock_requests[i], sem))
                for i in range(start_idx, end_idx)
            ]
            all_tasks.extend(step_tasks)
            
            if step < ramp_steps - 1:
                await asyncio.sleep(step_duration)
            
            if self.debug and self._debug_logger:
                step_elapsed = time.perf_counter() - step_start
                self._debug_logger.step_complete(
                    step + 1, ramp_steps, step_requests, step_elapsed
                )
        
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        if profile.hold_time > 0:
            print(f"   ⏸️  Holding at target load for {profile.hold_time}s...")
            if self.debug and self._debug_logger:
                self._debug_logger.mode_phase("HOLD", f"duration={profile.hold_time}s")
            await asyncio.sleep(profile.hold_time)


class SteppingRunner(ModeRunner):
    """
    JMeter Stepping Thread Group pattern.
    
    Adds users in discrete steps with configurable step size and duration.
    """
    
    async def run(
        self,
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker: Callable,
        profile: LoadProfile
    ):
        total = len(mock_requests)
        target_concurrency = scenario.concurrency
        num_steps = profile.ramp_up_steps if profile.ramp_up_steps > 1 else 5
        step_hold_time = profile.hold_time if profile.hold_time > 0 else 2.0
        
        threads_per_step = max(1, target_concurrency // num_steps)
        requests_per_step = total // num_steps
        
        print(f"   📶 Stepping Thread Group: {num_steps} steps, +{threads_per_step} threads/step, {step_hold_time}s hold")
        
        if self.debug and self._debug_logger:
            self._debug_logger.mode_phase(
                "STEPPING", 
                f"steps={num_steps}, threads_per_step={threads_per_step}, hold={step_hold_time}s"
            )
        
        all_tasks = []
        current_concurrency = 0
        
        for step in range(num_steps):
            step_start = time.perf_counter()
            current_concurrency = min(current_concurrency + threads_per_step, target_concurrency)
            start_idx = step * requests_per_step
            end_idx = start_idx + requests_per_step if step < num_steps - 1 else total
            step_requests = end_idx - start_idx
            
            print(f"   Step {step + 1}/{num_steps}: {current_concurrency} concurrent threads")
            
            if self.debug and self._debug_logger:
                self._debug_logger.step_start(
                    step + 1, num_steps, current_concurrency, 
                    step_requests, step_hold_time
                )
            
            sem = asyncio.Semaphore(current_concurrency)
            step_tasks = [
                asyncio.create_task(worker(i, mock_requests[i], sem))
                for i in range(start_idx, end_idx)
            ]
            all_tasks.extend(step_tasks)
            
            await asyncio.sleep(step_hold_time)
            
            if self.debug and self._debug_logger:
                step_elapsed = time.perf_counter() - step_start
                self._debug_logger.step_complete(
                    step + 1, num_steps, step_requests, step_elapsed
                )
        
        await asyncio.gather(*all_tasks, return_exceptions=True)


class SpikeRunner(ModeRunner):
    """
    Spike testing pattern.
    
    Sudden burst of traffic:
    1. Normal load phase
    2. Spike phase (multiplied concurrency)
    3. Return to normal
    """
    
    async def run(
        self,
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker: Callable,
        profile: LoadProfile
    ):
        total = len(mock_requests)
        base_concurrency = scenario.concurrency
        spike_concurrency = int(base_concurrency * profile.spike_multiplier)
        spike_duration = profile.spike_duration
        
        # Distribute requests: 40% pre-spike, 30% spike, 30% post-spike
        pre_spike_requests = int(total * 0.4)
        spike_requests = int(total * 0.3)
        post_spike_requests = total - pre_spike_requests - spike_requests
        
        print(f"   ⚡ Spike test: {base_concurrency} → {spike_concurrency} → {base_concurrency}")
        print(f"      Pre-spike: {pre_spike_requests} requests at {base_concurrency} concurrency")
        print(f"      Spike: {spike_requests} requests at {spike_concurrency} concurrency for {spike_duration}s")
        print(f"      Post-spike: {post_spike_requests} requests at {base_concurrency} concurrency")
        
        if self.debug and self._debug_logger:
            self._debug_logger.mode_phase(
                "SPIKE", 
                f"base={base_concurrency}, spike={spike_concurrency}, multiplier={profile.spike_multiplier}x"
            )
        
        # Phase 1: Pre-spike (normal load)
        print(f"   Phase 1: Pre-spike load...")
        if self.debug and self._debug_logger:
            self._debug_logger.spike_phase("PRE-SPIKE", base_concurrency, 0)
        
        phase1_start = time.perf_counter()
        sem = asyncio.Semaphore(base_concurrency)
        pre_tasks = [
            asyncio.create_task(worker(i, mock_requests[i], sem))
            for i in range(pre_spike_requests)
        ]
        await asyncio.gather(*pre_tasks, return_exceptions=True)
        
        if self.debug and self._debug_logger:
            phase1_elapsed = time.perf_counter() - phase1_start
            self._debug_logger.mode_phase(
                "PRE-SPIKE COMPLETE", 
                f"requests={pre_spike_requests}, elapsed={phase1_elapsed:.2f}s"
            )
        
        # Phase 2: Spike
        print(f"   Phase 2: SPIKE! ({spike_concurrency} concurrent)")
        if self.debug and self._debug_logger:
            self._debug_logger.spike_phase("SPIKE", spike_concurrency, spike_duration)
        
        spike_start = time.perf_counter()
        spike_sem = asyncio.Semaphore(spike_concurrency)
        spike_tasks = [
            asyncio.create_task(
                worker(
                    i + pre_spike_requests, 
                    mock_requests[i + pre_spike_requests], 
                    spike_sem
                )
            )
            for i in range(spike_requests)
        ]
        
        await asyncio.gather(*spike_tasks, return_exceptions=True)
        spike_elapsed = time.perf_counter() - spike_start
        if spike_elapsed < spike_duration:
            remaining = spike_duration - spike_elapsed
            if self.debug and self._debug_logger:
                self._debug_logger.mode_phase("SPIKE HOLD", f"waiting {remaining:.2f}s more")
            await asyncio.sleep(remaining)
        
        if self.debug and self._debug_logger:
            total_spike_time = time.perf_counter() - spike_start
            self._debug_logger.mode_phase(
                "SPIKE COMPLETE", 
                f"requests={spike_requests}, elapsed={total_spike_time:.2f}s"
            )
        
        # Phase 3: Post-spike (return to normal)
        print(f"   Phase 3: Recovery...")
        if self.debug and self._debug_logger:
            self._debug_logger.spike_phase("RECOVERY", base_concurrency, 0)
        
        phase3_start = time.perf_counter()
        offset = pre_spike_requests + spike_requests
        post_tasks = [
            asyncio.create_task(worker(i + offset, mock_requests[i + offset], sem))
            for i in range(post_spike_requests)
        ]
        await asyncio.gather(*post_tasks, return_exceptions=True)
        
        if self.debug and self._debug_logger:
            phase3_elapsed = time.perf_counter() - phase3_start
            self._debug_logger.mode_phase(
                "RECOVERY COMPLETE", 
                f"requests={post_spike_requests}, elapsed={phase3_elapsed:.2f}s"
            )


class ConstantRateRunner(ModeRunner):
    """
    Constant Throughput pattern (JMeter Constant Throughput Timer).
    
    Maintains a fixed requests-per-second rate regardless of response times.
    """
    
    async def run(
        self,
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker: Callable,
        profile: LoadProfile
    ):
        total = len(mock_requests)
        target_rps = profile.target_rps if profile.target_rps > 0 else float(scenario.concurrency)
        interval = 1.0 / target_rps
        
        print(f"   🎯 Constant rate: {target_rps:.2f} requests/second")
        print(f"      Interval: {interval * 1000:.2f}ms between requests")
        
        if self.debug and self._debug_logger:
            self._debug_logger.mode_phase(
                "CONSTANT_RATE", 
                f"target_rps={target_rps:.2f}, interval={interval*1000:.2f}ms"
            )
        
        all_tasks = []
        sem = asyncio.Semaphore(scenario.concurrency)
        
        start_time = time.perf_counter()
        last_rate_log = start_time
        requests_since_log = 0
        
        for i, mock_request in enumerate(mock_requests):
            expected_time = i * interval
            current_time = time.perf_counter() - start_time
            
            if current_time < expected_time:
                await asyncio.sleep(expected_time - current_time)
            
            task = asyncio.create_task(worker(i, mock_request, sem))
            all_tasks.append(task)
            requests_since_log += 1
            
            if self.debug and self._debug_logger and (time.perf_counter() - last_rate_log) >= 1.0:
                actual_rps = requests_since_log / (time.perf_counter() - last_rate_log)
                adjustment = (1.0 / actual_rps - interval) * 1000 if actual_rps > 0 else 0
                self._debug_logger.rate_limit(target_rps, actual_rps, adjustment)
                last_rate_log = time.perf_counter()
                requests_since_log = 0
        
        await asyncio.gather(*all_tasks, return_exceptions=True)


class ArrivalsRunner(ModeRunner):
    """
    Arrivals Thread Group pattern (JMeter Arrivals Thread Group).
    
    Controls the rate at which new requests arrive, regardless of completion.
    Uses Poisson distribution for realistic arrival patterns.
    """
    
    async def run(
        self,
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker: Callable,
        profile: LoadProfile
    ):
        total = len(mock_requests)
        arrival_rate = profile.arrival_rate if profile.arrival_rate > 0 else 10.0
        
        print(f"   🚪 Arrivals mode: {arrival_rate:.2f} arrivals/second")
        print(f"      No concurrency limit - all arrivals queued immediately")
        
        if self.debug and self._debug_logger:
            self._debug_logger.mode_phase(
                "ARRIVALS", 
                f"arrival_rate={arrival_rate:.2f}/sec, total={total}"
            )
        
        all_tasks = []
        
        start_time = time.perf_counter()
        last_arrival_log = start_time
        arrivals_since_log = 0
        
        for i, mock_request in enumerate(mock_requests):
            expected_time = i / arrival_rate
            current_time = time.perf_counter() - start_time
            
            if current_time < expected_time:
                jitter = random.expovariate(arrival_rate) * 0.1
                await asyncio.sleep(max(0, expected_time - current_time + jitter))
            
            task = asyncio.create_task(worker(i, mock_request))
            all_tasks.append(task)
            arrivals_since_log += 1
            
            if self.debug and self._debug_logger and (time.perf_counter() - last_arrival_log) >= 1.0:
                actual_rate = arrivals_since_log / (time.perf_counter() - last_arrival_log)
                self._debug_logger.rate_limit(arrival_rate, actual_rate, 0)
                last_arrival_log = time.perf_counter()
                arrivals_since_log = 0
        
        await asyncio.gather(*all_tasks, return_exceptions=True)


class UltimateRunner(ModeRunner):
    """
    Ultimate Thread Group pattern (JMeter Ultimate Thread Group).
    
    Complex multi-phase load pattern with stages.
    """
    
    async def run(
        self,
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker: Callable,
        profile: LoadProfile
    ):
        if not profile.stages:
            target = scenario.concurrency
            profile.stages = [
                (0, target // 3, 5.0),
                (target // 3, target, 5.0),
                (target, target, 10.0),
                (target, target // 2, 5.0),
                (target // 2, 0, 5.0),
            ]
        
        total_stages = len(profile.stages)
        total_stage_duration = sum(s[2] for s in profile.stages)
        
        print(f"   🎛️  Ultimate Thread Group: {total_stages} stages over {total_stage_duration}s")
        for i, (start, end, duration) in enumerate(profile.stages):
            direction = "→" if start < end else ("←" if start > end else "=")
            print(f"      Stage {i+1}: {start} {direction} {end} threads over {duration}s")
        
        if self.debug and self._debug_logger:
            self._debug_logger.mode_phase(
                "ULTIMATE", 
                f"stages={total_stages}, total_duration={total_stage_duration}s"
            )
        
        total = len(mock_requests)
        requests_per_stage = total // total_stages
        all_tasks = []
        request_idx = 0
        
        for stage_idx, (start_threads, end_threads, duration) in enumerate(profile.stages):
            stage_requests = requests_per_stage if stage_idx < total_stages - 1 else (total - request_idx)
            
            print(f"   Executing stage {stage_idx + 1}...")
            
            if self.debug and self._debug_logger:
                self._debug_logger.stage_start(
                    stage_idx + 1, total_stages,
                    start_threads, end_threads, duration
                )
            
            sub_steps = max(1, int(duration))
            requests_per_sub = max(1, stage_requests // sub_steps)
            
            stage_start_time = time.perf_counter()
            
            for sub in range(sub_steps):
                progress = sub / max(1, sub_steps - 1)
                current_threads = int(start_threads + (end_threads - start_threads) * progress)
                current_threads = max(1, current_threads)
                
                sub_request_count = requests_per_sub if sub < sub_steps - 1 else (stage_requests - sub * requests_per_sub)
                
                if self.debug and self._debug_logger and sub % 5 == 0:
                    elapsed = time.perf_counter() - stage_start_time
                    self._debug_logger.mode_phase(
                        f"STAGE {stage_idx + 1} SUB {sub + 1}/{sub_steps}",
                        f"threads={current_threads}, elapsed={elapsed:.1f}s"
                    )
                
                sem = asyncio.Semaphore(current_threads)
                sub_tasks = [
                    asyncio.create_task(worker(request_idx + j, mock_requests[request_idx + j], sem))
                    for j in range(min(sub_request_count, total - request_idx))
                ]
                all_tasks.extend(sub_tasks)
                request_idx += len(sub_tasks)
                
                await asyncio.sleep(1.0)
            
            if self.debug and self._debug_logger:
                self._debug_logger.mode_phase(f"STAGE {stage_idx + 1} COMPLETE", "")
        
        await asyncio.gather(*all_tasks, return_exceptions=True)


class DurationRunner(ModeRunner):
    """
    Duration-based execution (run for fixed time, cycling through requests).
    """
    
    async def run(
        self,
        mock_requests: List[MockRequest],
        scenario: ScenarioConfig,
        worker: Callable,
        profile: LoadProfile
    ):
        duration = profile.duration_seconds if profile.duration_seconds > 0 else 60.0
        concurrency = scenario.concurrency
        
        print(f"   ⏱️  Duration mode: Running for {duration}s at {concurrency} concurrency")
        print(f"      Requests will cycle if duration exceeds request count")
        
        if self.debug and self._debug_logger:
            self._debug_logger.mode_phase(
                "DURATION", 
                f"duration={duration}s, concurrency={concurrency}"
            )
        
        all_tasks = []
        sem = asyncio.Semaphore(concurrency)
        
        start_time = time.perf_counter()
        request_idx = 0
        requests_sent = 0
        last_progress_log = start_time
        
        while (time.perf_counter() - start_time) < duration:
            mock_request = mock_requests[request_idx % len(mock_requests)]
            
            task = asyncio.create_task(worker(requests_sent, mock_request, sem))
            all_tasks.append(task)
            
            request_idx += 1
            requests_sent += 1
            
            if self.debug and self._debug_logger and (time.perf_counter() - last_progress_log) >= 5.0:
                elapsed = time.perf_counter() - start_time
                self._debug_logger.duration_progress(
                    elapsed, duration, requests_sent, concurrency
                )
                last_progress_log = time.perf_counter()
            
            await asyncio.sleep(0.001)
        
        print(f"      Sent {requests_sent} requests in {duration}s")
        
        if self.debug and self._debug_logger:
            self._debug_logger.mode_phase(
                "DURATION COMPLETE", 
                f"total_sent={requests_sent}, duration={duration}s"
            )
        
        await asyncio.gather(*all_tasks, return_exceptions=True)


def get_mode_runner(
    mode: BenchmarkMode, 
    debug: bool = False, 
    debug_logger: Optional[DebugLogger] = None
) -> Optional[ModeRunner]:
    """Get the appropriate runner for a benchmark mode."""
    runners = {
        BenchmarkMode.RAMP_UP: RampUpRunner,
        BenchmarkMode.STEPPING: SteppingRunner,
        BenchmarkMode.SPIKE: SpikeRunner,
        BenchmarkMode.CONSTANT_RATE: ConstantRateRunner,
        BenchmarkMode.ARRIVALS: ArrivalsRunner,
        BenchmarkMode.ULTIMATE: UltimateRunner,
        BenchmarkMode.DURATION: DurationRunner,
    }
    
    runner_class = runners.get(mode)
    if runner_class:
        return runner_class(debug=debug, debug_logger=debug_logger)
    return None
