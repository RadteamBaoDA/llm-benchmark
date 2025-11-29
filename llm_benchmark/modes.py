"""
Benchmark modes and load profile configurations.

Based on JMeter Thread Group specifications for comprehensive load testing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class BenchmarkMode(Enum):
    """
    Benchmark execution modes based on JMeter Thread Group patterns.
    
    JMeter-inspired modes:
    - PARALLEL: Blazemeter-style instant load (all threads at once)
    - CONTROLLED: Standard Thread Group with concurrency limit
    - RAMP_UP: Linear ramp-up like JMeter's ramp-up period
    - STEPPING: Stepping Thread Group - add users in steps
    - SPIKE: Spike testing - sudden burst then sustain
    - CONSTANT_RATE: Constant Throughput Timer - fixed RPS
    - ARRIVALS: Arrivals Thread Group - control arrival rate
    - ULTIMATE: Ultimate Thread Group - complex load pattern
    - QUEUE_TEST: Queue depth testing for inference servers
    - DURATION: Run for fixed duration instead of request count
    """
    PARALLEL = "parallel"
    CONTROLLED = "controlled"
    RAMP_UP = "ramp_up"
    STEPPING = "stepping"
    SPIKE = "spike"
    CONSTANT_RATE = "constant_rate"
    ARRIVALS = "arrivals"
    ULTIMATE = "ultimate"
    QUEUE_TEST = "queue_test"
    DURATION = "duration"
    
    @classmethod
    def from_string(cls, value: str) -> "BenchmarkMode":
        """Create BenchmarkMode from string value."""
        value_lower = value.lower()
        for mode in cls:
            if mode.value == value_lower:
                return mode
        raise ValueError(f"Unknown benchmark mode: {value}")


@dataclass
class LoadProfile:
    """
    Load profile configuration for advanced execution modes.
    Based on JMeter Ultimate Thread Group patterns.
    """
    # Ramp-up settings
    ramp_up_time: float = 0.0  # seconds to reach target concurrency
    ramp_up_steps: int = 1  # number of steps for stepping mode
    
    # Hold/sustain settings
    hold_time: float = 0.0  # seconds to hold at target load
    
    # Ramp-down settings
    ramp_down_time: float = 0.0  # seconds to ramp down
    
    # Constant rate settings
    target_rps: float = 0.0  # requests per second for constant rate mode
    
    # Spike settings
    spike_multiplier: float = 2.0  # multiply concurrency during spike
    spike_duration: float = 5.0  # seconds for spike
    
    # Arrivals settings  
    arrival_rate: float = 10.0  # new requests per second
    
    # Duration mode
    duration_seconds: float = 60.0  # total test duration
    
    # Ultimate Thread Group pattern (list of stages)
    # Each stage: (start_threads, end_threads, duration_seconds)
    stages: List[Tuple[int, int, float]] = field(default_factory=list)
    
    @classmethod
    def from_scenario(cls, scenario) -> "LoadProfile":
        """Extract load profile from scenario configuration."""
        profile = cls()
        
        # Map scenario attributes to profile fields
        attr_mapping = {
            'ramp_up_time': 'ramp_up_time',
            'ramp_up_steps': 'ramp_up_steps',
            'hold_time': 'hold_time',
            'ramp_down_time': 'ramp_down_time',
            'target_rps': 'target_rps',
            'spike_multiplier': 'spike_multiplier',
            'spike_duration': 'spike_duration',
            'arrival_rate': 'arrival_rate',
            'duration_seconds': 'duration_seconds',
            'stages': 'stages',
        }
        
        for scenario_attr, profile_attr in attr_mapping.items():
            if hasattr(scenario, scenario_attr):
                setattr(profile, profile_attr, getattr(scenario, scenario_attr))
                
        return profile


@dataclass
class QueueMetrics:
    """Metrics for queue behavior analysis."""
    queue_depth_samples: List[Tuple[float, int]] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    rejection_count: int = 0
    timeout_count: int = 0
    max_observed_queue_depth: int = 0
    avg_queue_depth: float = 0.0
    
    def calculate(self):
        """Calculate queue statistics."""
        if self.queue_depth_samples:
            depths = [d for _, d in self.queue_depth_samples]
            self.max_observed_queue_depth = max(depths) if depths else 0
            self.avg_queue_depth = sum(depths) / len(depths) if depths else 0
    
    def record_rejection(self):
        """Record a queue rejection (429/503)."""
        self.rejection_count += 1
    
    def record_timeout(self):
        """Record a timeout."""
        self.timeout_count += 1
    
    def add_queue_sample(self, elapsed: float, depth: int):
        """Add a queue depth sample."""
        self.queue_depth_samples.append((elapsed, depth))
    
    def add_wait_time(self, wait_time: float):
        """Add a wait time sample."""
        if wait_time > 0.001:  # Only record significant waits
            self.wait_times.append(wait_time)
    
    def add_processing_time(self, processing_time: float):
        """Add a processing time sample."""
        if processing_time:
            self.processing_times.append(processing_time)
