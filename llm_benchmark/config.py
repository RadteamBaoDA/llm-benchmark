"""
Configuration loader and validator for LLM Benchmark Tool.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import yaml


@dataclass
class APIConfig:
    """API configuration settings."""
    base_url: str
    api_key: str = ""
    timeout: int = 60


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    type: str  # chat, embed, reranker, vision
    max_tokens: int = 32
    temperature: float = 0.2


@dataclass
class MockDataConfig:
    """Mock data configuration for different model types."""
    # Chat model
    chat_prompts: List[str] = field(default_factory=lambda: [
        "Hello, world!",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of exercise?"
    ])
    
    # Embed model
    embed_texts: List[str] = field(default_factory=lambda: [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of data."
    ])
    
    # Reranker model
    reranker_query: str = "What is machine learning?"
    reranker_documents: List[str] = field(default_factory=lambda: [
        "Machine learning is a type of artificial intelligence that allows computers to learn from data.",
        "The weather today is sunny with a high of 75 degrees.",
        "Python is commonly used for machine learning applications.",
        "Machine learning algorithms can identify patterns in large datasets.",
        "Coffee is one of the most popular beverages in the world."
    ])
    
    # Vision model
    vision_prompts: List[str] = field(default_factory=lambda: [
        "Describe this image in detail.",
        "What objects can you see in this image?",
        "What is the main subject of this image?"
    ])
    vision_image_url: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    vision_image_base64: Optional[str] = None
    vision_image_path: Optional[str] = None  # Single local image path
    vision_image_paths: List[str] = field(default_factory=list)  # Multiple local image paths


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_requests: bool = False  # Log HTTP request details (URL, headers, payload)
    log_responses: bool = False  # Log HTTP response details (status, body)
    log_file: Optional[str] = None  # Log file path (None = console only)
    max_payload_length: int = 500  # Truncate payload in logs
    max_response_length: int = 500  # Truncate response body in logs


@dataclass
class ScenarioConfig:
    """
    Benchmark scenario configuration.
    
    Supports JMeter-style execution modes:
    - parallel: All requests launched simultaneously
    - controlled: Semaphore-limited concurrency
    - queue_test: Queue depth testing for inference servers
    - ramp_up: Linear ramp-up (JMeter ramp-up period)
    - stepping: JMeter Stepping Thread Group
    - spike: Sudden burst testing
    - constant_rate: JMeter Constant Throughput Timer
    - arrivals: JMeter Arrivals Thread Group
    - ultimate: JMeter Ultimate Thread Group (multi-phase)
    - duration: Run for fixed time
    """
    name: str
    requests: int
    concurrency: int
    description: str = ""
    warmup_requests: int = 1
    timeout: int = 60
    enabled: bool = True
    mode: str = "parallel"
    
    # JMeter-style load profile parameters
    ramp_up_time: float = 0.0  # Seconds to reach target concurrency
    ramp_up_steps: int = 1  # Number of steps for stepping/ramp-up
    hold_time: float = 0.0  # Seconds to hold at target load
    ramp_down_time: float = 0.0  # Seconds to ramp down
    
    # Constant rate / throughput settings
    target_rps: float = 0.0  # Target requests per second
    
    # Spike settings
    spike_multiplier: float = 2.0  # Multiply concurrency during spike
    spike_duration: float = 5.0  # Duration of spike in seconds
    
    # Arrivals settings
    arrival_rate: float = 10.0  # New requests per second
    
    # Duration mode
    duration_seconds: float = 60.0  # Total test duration
    
    # Ultimate Thread Group stages: list of (start_threads, end_threads, duration)
    stages: List[tuple] = field(default_factory=list)


@dataclass
class ScenarioDefaults:
    """Default settings for scenarios."""
    requests: int = 100
    concurrency: int = 10
    warmup_requests: int = 1
    timeout: int = 60
    enabled: bool = True
    mode: str = "parallel"
    
    # JMeter-style load profile defaults
    ramp_up_time: float = 0.0
    ramp_up_steps: int = 1
    hold_time: float = 0.0
    ramp_down_time: float = 0.0
    target_rps: float = 0.0
    spike_multiplier: float = 2.0
    spike_duration: float = 5.0
    arrival_rate: float = 10.0
    duration_seconds: float = 60.0


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    api: APIConfig
    model: ModelConfig
    mock_data: MockDataConfig = field(default_factory=MockDataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    scenarios: List[ScenarioConfig] = field(default_factory=list)
    scenario_file: str = "scenario.yml"
    scenario_defaults: ScenarioDefaults = field(default_factory=ScenarioDefaults)
    default_requests: int = 100
    default_concurrency: int = 10
    capture_responses: bool = False
    output_dir: str = "results"
    export_formats: List[str] = field(default_factory=lambda: ["markdown", "csv"])
    quiet: bool = False


def load_config(config_path: str = "config.yml") -> BenchmarkConfig:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    # Get config directory for relative paths
    config_dir = config_file.parent
    
    return parse_config(raw_config, config_dir)


def load_scenarios(scenario_path: Path, defaults: ScenarioDefaults) -> List[ScenarioConfig]:
    """Load scenarios from a separate YAML file."""
    if not scenario_path.exists():
        return []
    
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario_data = yaml.safe_load(f)
    
    if not scenario_data:
        return []
    
    # Get defaults from scenario file or use provided defaults
    file_defaults = scenario_data.get('defaults', {})
    defaults = ScenarioDefaults(
        requests=file_defaults.get('requests', defaults.requests),
        concurrency=file_defaults.get('concurrency', defaults.concurrency),
        warmup_requests=file_defaults.get('warmup_requests', defaults.warmup_requests),
        timeout=file_defaults.get('timeout', defaults.timeout),
        enabled=file_defaults.get('enabled', defaults.enabled),
        mode=file_defaults.get('mode', defaults.mode),
        ramp_up_time=file_defaults.get('ramp_up_time', defaults.ramp_up_time),
        ramp_up_steps=file_defaults.get('ramp_up_steps', defaults.ramp_up_steps),
        hold_time=file_defaults.get('hold_time', defaults.hold_time),
        ramp_down_time=file_defaults.get('ramp_down_time', defaults.ramp_down_time),
        target_rps=file_defaults.get('target_rps', defaults.target_rps),
        spike_multiplier=file_defaults.get('spike_multiplier', defaults.spike_multiplier),
        spike_duration=file_defaults.get('spike_duration', defaults.spike_duration),
        arrival_rate=file_defaults.get('arrival_rate', defaults.arrival_rate),
        duration_seconds=file_defaults.get('duration_seconds', defaults.duration_seconds),
    )
    
    scenarios = []
    for s in scenario_data.get('scenarios', []):
        # Parse stages if present (for Ultimate Thread Group)
        stages_raw = s.get('stages', [])
        stages = [(stage.get('start', 0), stage.get('end', 0), stage.get('duration', 0)) 
                  for stage in stages_raw] if stages_raw else []
        
        scenario = ScenarioConfig(
            name=s.get('name', 'default'),
            requests=s.get('requests', defaults.requests),
            concurrency=s.get('concurrency', defaults.concurrency),
            description=s.get('description', ''),
            warmup_requests=s.get('warmup_requests', defaults.warmup_requests),
            timeout=s.get('timeout', defaults.timeout),
            enabled=s.get('enabled', defaults.enabled),
            mode=s.get('mode', defaults.mode),
            ramp_up_time=s.get('ramp_up_time', defaults.ramp_up_time),
            ramp_up_steps=s.get('ramp_up_steps', defaults.ramp_up_steps),
            hold_time=s.get('hold_time', defaults.hold_time),
            ramp_down_time=s.get('ramp_down_time', defaults.ramp_down_time),
            target_rps=s.get('target_rps', defaults.target_rps),
            spike_multiplier=s.get('spike_multiplier', defaults.spike_multiplier),
            spike_duration=s.get('spike_duration', defaults.spike_duration),
            arrival_rate=s.get('arrival_rate', defaults.arrival_rate),
            duration_seconds=s.get('duration_seconds', defaults.duration_seconds),
            stages=stages,
        )
        scenarios.append(scenario)
    
    return scenarios


def parse_config(raw_config: Dict[str, Any], config_dir: Optional[Path] = None) -> BenchmarkConfig:
    """Parse raw configuration dictionary into BenchmarkConfig."""
    if config_dir is None:
        config_dir = Path.cwd()
    
    # Parse API config
    """Parse raw configuration dictionary into BenchmarkConfig."""
    # Parse API config
    api_raw = raw_config.get('api', {})
    api_config = APIConfig(
        base_url=api_raw.get('base_url', 'http://localhost:8000'),
        api_key=api_raw.get('api_key', os.environ.get('OPENAI_API_KEY', '')),
        timeout=api_raw.get('timeout', 60)
    )
    
    # Parse Model config
    model_raw = raw_config.get('model', {})
    model_config = ModelConfig(
        name=model_raw.get('name', 'gpt-3.5-turbo'),
        type=model_raw.get('type', 'chat'),
        max_tokens=model_raw.get('max_tokens', 32),
        temperature=model_raw.get('temperature', 0.2)
    )
    
    # Parse Mock data config - create defaults first, then override
    mock_raw = raw_config.get('mock_data', {})
    default_mock = MockDataConfig()
    mock_config = MockDataConfig(
        chat_prompts=mock_raw.get('chat_prompts', default_mock.chat_prompts),
        embed_texts=mock_raw.get('embed_texts', default_mock.embed_texts),
        reranker_query=mock_raw.get('reranker_query', default_mock.reranker_query),
        reranker_documents=mock_raw.get('reranker_documents', default_mock.reranker_documents),
        vision_prompts=mock_raw.get('vision_prompts', default_mock.vision_prompts),
        vision_image_url=mock_raw.get('vision_image_url', default_mock.vision_image_url),
        vision_image_base64=mock_raw.get('vision_image_base64', None),
        vision_image_path=mock_raw.get('vision_image_path', None),
        vision_image_paths=mock_raw.get('vision_image_paths', [])
    )
    
    # Parse Logging config
    logging_raw = raw_config.get('logging', {})
    logging_config = LoggingConfig(
        level=logging_raw.get('level', 'INFO'),
        log_requests=logging_raw.get('log_requests', False),
        log_responses=logging_raw.get('log_responses', False),
        log_file=logging_raw.get('log_file', None),
        max_payload_length=logging_raw.get('max_payload_length', 500),
        max_response_length=logging_raw.get('max_response_length', 500)
    )
    
    # Parse Scenarios - from separate file or inline
    scenario_file = raw_config.get('scenario_file', 'scenario.yml')
    scenario_defaults = ScenarioDefaults()
    
    # Check for scenario file
    scenario_path = config_dir / scenario_file
    if scenario_path.exists():
        scenarios = load_scenarios(scenario_path, scenario_defaults)
    else:
        # Fall back to inline scenarios if present
        scenarios_raw = raw_config.get('scenarios', [])
        scenarios = []
        for s in scenarios_raw:
            scenarios.append(ScenarioConfig(
                name=s.get('name', 'default'),
                requests=s.get('requests', scenario_defaults.requests),
                concurrency=s.get('concurrency', scenario_defaults.concurrency),
                description=s.get('description', ''),
                warmup_requests=s.get('warmup_requests', scenario_defaults.warmup_requests),
                timeout=s.get('timeout', scenario_defaults.timeout),
                enabled=s.get('enabled', scenario_defaults.enabled),
                mode=s.get('mode', scenario_defaults.mode)
            ))
    
    # Parse benchmark settings
    benchmark_raw = raw_config.get('benchmark', {})
    
    return BenchmarkConfig(
        api=api_config,
        model=model_config,
        mock_data=mock_config,
        logging=logging_config,
        scenarios=scenarios,
        scenario_file=scenario_file,
        scenario_defaults=scenario_defaults,
        default_requests=benchmark_raw.get('default_requests', 100),
        default_concurrency=benchmark_raw.get('default_concurrency', 10),
        capture_responses=benchmark_raw.get('capture_responses', False),
        output_dir=benchmark_raw.get('output_dir', 'results'),
        export_formats=benchmark_raw.get('export_formats', ['markdown', 'csv']),
        quiet=benchmark_raw.get('quiet', False)
    )


def create_default_config() -> str:
    """Create a default configuration YAML string."""
    return """# LLM Benchmark Tool Configuration

# API Configuration
api:
  base_url: "http://localhost:8000"
  api_key: ""  # Or use environment variable OPENAI_API_KEY
  timeout: 60

# Model Configuration
model:
  name: "gpt-3.5-turbo"
  type: "chat"  # Options: chat, embed, reranker, vision
  max_tokens: 32
  temperature: 0.2

# Mock Data Configuration (used for generating benchmark requests)
mock_data:
  # Chat model prompts
  chat_prompts:
    - "Hello, world!"
    - "What is the capital of France?"
    - "Explain quantum computing in simple terms."
    - "Write a haiku about programming."
    - "What are the benefits of exercise?"
  
  # Embedding texts
  embed_texts:
    - "The quick brown fox jumps over the lazy dog."
    - "Machine learning is a subset of artificial intelligence."
    - "Python is a popular programming language."
    - "Natural language processing enables computers to understand text."
    - "Deep learning models require large amounts of data."
  
  # Reranker configuration
  reranker_query: "What is machine learning?"
  reranker_documents:
    - "Machine learning is a type of artificial intelligence that allows computers to learn from data."
    - "The weather today is sunny with a high of 75 degrees."
    - "Python is commonly used for machine learning applications."
    - "Machine learning algorithms can identify patterns in large datasets."
    - "Coffee is one of the most popular beverages in the world."
  
  # Vision model configuration
  vision_prompts:
    - "Describe this image in detail."
    - "What objects can you see in this image?"
    - "What is the main subject of this image?"
  vision_image_url: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
  # vision_image_base64: null  # Optional: base64 encoded image
  # vision_image_path: null  # Optional: single local image path (e.g., "./images/test.jpg")
  # vision_image_paths:  # Optional: multiple local image paths (cycles through them)
  #   - "./images/image1.jpg"
  #   - "./images/image2.png"
  #   - "./images/image3.webp"

# Logging Configuration
# Enable debug logging to diagnose request failures
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_requests: false  # Log HTTP request details (URL, headers, payload)
  log_responses: false  # Log HTTP response details (status, body)
  log_file: null  # Log file path (null = console only, e.g., "benchmark.log")
  max_payload_length: 500  # Truncate payload in logs (characters)
  max_response_length: 500  # Truncate response body in logs (characters)

# Scenario Configuration
# Scenarios are defined in a separate file for better organization
scenario_file: "scenario.yml"

# Benchmark Settings
benchmark:
  default_requests: 100
  default_concurrency: 10
  capture_responses: false
  output_dir: "results"
  export_formats:
    - "markdown"
    - "csv"
  quiet: false
"""


def create_default_scenario_config() -> str:
    """Create a default scenario configuration YAML string."""
    return """# LLM Benchmark Scenarios Configuration
# Define multiple benchmark scenarios with different load profiles
#
# ============================================================================
# EXECUTION MODES (Based on JMeter Thread Group Patterns)
# ============================================================================
#
# Basic Modes:
#   - parallel: All requests launched simultaneously (tests max queue pressure)
#   - controlled: Uses semaphore to limit concurrent requests
#   - queue_test: Burst mode to analyze queue behavior (Ollama/vLLM)
#
# JMeter-Style Modes:
#   - ramp_up: Linear ramp-up like JMeter's Thread Group ramp-up period
#       Parameters: ramp_up_time (seconds), ramp_up_steps, hold_time
#
#   - stepping: JMeter Stepping Thread Group pattern
#       Parameters: ramp_up_steps (number of steps), hold_time (per step)
#       Adds users in discrete steps with configurable hold time
#
#   - spike: Spike testing pattern
#       Parameters: spike_multiplier (2.0 = double), spike_duration (seconds)
#       Tests system response to sudden traffic bursts
#
#   - constant_rate: JMeter Constant Throughput Timer pattern
#       Parameters: target_rps (requests per second)
#       Maintains fixed request rate regardless of response time
#
#   - arrivals: JMeter Arrivals Thread Group pattern
#       Parameters: arrival_rate (arrivals per second)
#       Controls arrival rate, allows unlimited concurrent requests
#
#   - ultimate: JMeter Ultimate Thread Group pattern
#       Parameters: stages (list of {start, end, duration})
#       Complex multi-phase load patterns with ramp-up/hold/ramp-down
#
#   - duration: Duration-based execution
#       Parameters: duration_seconds
#       Runs for fixed time, cycling through requests as needed
#
# ============================================================================

scenarios:
  # -------------------------------------------------------------------------
  # BASIC VALIDATION SCENARIOS
  # -------------------------------------------------------------------------
  
  - name: "light_load"
    description: "Light load test for basic validation"
    requests: 50
    concurrency: 5
    warmup_requests: 2
    timeout: 60
    enabled: true
    mode: "controlled"

  - name: "medium_load"
    description: "Medium load test for standard benchmarking"
    requests: 100
    concurrency: 10
    warmup_requests: 3
    timeout: 60
    enabled: true
    mode: "parallel"

  # -------------------------------------------------------------------------
  # JMETER-STYLE RAMP-UP (Thread Group with ramp-up period)
  # -------------------------------------------------------------------------
  
  - name: "jmeter_ramp_up"
    description: "JMeter-style linear ramp-up to target concurrency"
    requests: 200
    concurrency: 40
    warmup_requests: 2
    timeout: 120
    enabled: false
    mode: "ramp_up"
    ramp_up_time: 30.0      # 30 seconds to reach full concurrency
    ramp_up_steps: 10       # 10 steps during ramp-up
    hold_time: 60.0         # Hold at target load for 60 seconds

  # -------------------------------------------------------------------------
  # JMETER STEPPING THREAD GROUP
  # -------------------------------------------------------------------------
  
  - name: "stepping_load"
    description: "JMeter Stepping Thread Group - add users in steps"
    requests: 300
    concurrency: 50
    warmup_requests: 2
    timeout: 120
    enabled: false
    mode: "stepping"
    ramp_up_steps: 5        # 5 steps: +10 users each step
    hold_time: 10.0         # Hold for 10 seconds at each step

  # -------------------------------------------------------------------------
  # SPIKE TESTING
  # -------------------------------------------------------------------------
  
  - name: "spike_test"
    description: "Spike test - sudden burst of traffic"
    requests: 200
    concurrency: 20         # Base concurrency
    warmup_requests: 3
    timeout: 120
    enabled: false
    mode: "spike"
    spike_multiplier: 5.0   # 5x spike (20 -> 100 concurrent)
    spike_duration: 10.0    # Spike lasts 10 seconds

  # -------------------------------------------------------------------------
  # CONSTANT THROUGHPUT (JMeter Constant Throughput Timer)
  # -------------------------------------------------------------------------
  
  - name: "constant_throughput"
    description: "Maintain fixed requests per second rate"
    requests: 300
    concurrency: 30         # Max concurrent requests allowed
    warmup_requests: 3
    timeout: 60
    enabled: false
    mode: "constant_rate"
    target_rps: 10.0        # Target: 10 requests per second

  # -------------------------------------------------------------------------
  # ARRIVALS THREAD GROUP (JMeter Arrivals Thread Group)
  # -------------------------------------------------------------------------
  
  - name: "arrivals_test"
    description: "Control arrival rate, unlimited concurrency"
    requests: 200
    concurrency: 100        # High limit for unbounded arrivals
    warmup_requests: 2
    timeout: 120
    enabled: false
    mode: "arrivals"
    arrival_rate: 20.0      # 20 new requests per second

  # -------------------------------------------------------------------------
  # ULTIMATE THREAD GROUP (JMeter Ultimate Thread Group)
  # -------------------------------------------------------------------------
  
  - name: "ultimate_load_pattern"
    description: "Complex multi-phase load pattern (ramp-up, hold, ramp-down)"
    requests: 500
    concurrency: 50
    warmup_requests: 5
    timeout: 180
    enabled: false
    mode: "ultimate"
    stages:
      - { start: 0, end: 10, duration: 10 }    # Ramp to 10 threads in 10s
      - { start: 10, end: 30, duration: 15 }   # Ramp to 30 threads in 15s
      - { start: 30, end: 50, duration: 10 }   # Ramp to 50 threads in 10s
      - { start: 50, end: 50, duration: 30 }   # Hold at 50 threads for 30s
      - { start: 50, end: 20, duration: 10 }   # Ramp down to 20 threads
      - { start: 20, end: 0, duration: 10 }    # Ramp down to 0 threads

  # -------------------------------------------------------------------------
  # DURATION-BASED TESTING
  # -------------------------------------------------------------------------
  
  - name: "duration_test"
    description: "Run for fixed duration, cycling through requests"
    requests: 100           # Template requests to cycle
    concurrency: 15
    warmup_requests: 3
    timeout: 60
    enabled: false
    mode: "duration"
    duration_seconds: 120.0  # Run for 2 minutes

  # -------------------------------------------------------------------------
  # QUEUE DEPTH TESTING (for Ollama/vLLM/TGI)
  # -------------------------------------------------------------------------
  
  - name: "queue_depth_test"
    description: "Test inference server queue depth and handling"
    requests: 100
    concurrency: 100        # All requests at once
    warmup_requests: 1
    timeout: 180
    enabled: true
    mode: "queue_test"

  # -------------------------------------------------------------------------
  # STRESS TESTING
  # -------------------------------------------------------------------------
  
  - name: "stress_test"
    description: "Stress test with high concurrency"
    requests: 500
    concurrency: 50
    warmup_requests: 5
    timeout: 120
    enabled: true
    mode: "parallel"

  # -------------------------------------------------------------------------
  # ENDURANCE TESTING
  # -------------------------------------------------------------------------
  
  - name: "endurance_test"
    description: "Long-running endurance test"
    requests: 1000
    concurrency: 10
    warmup_requests: 10
    timeout: 300
    enabled: false
    mode: "controlled"

# Default scenario settings (applied if not specified per scenario)
defaults:
  requests: 100
  concurrency: 10
  warmup_requests: 1
  timeout: 60
  enabled: true
  mode: "parallel"
  ramp_up_time: 0.0
  ramp_up_steps: 1
  hold_time: 0.0
  target_rps: 0.0
  spike_multiplier: 2.0
  spike_duration: 5.0
  arrival_rate: 10.0
  duration_seconds: 60.0
"""


def save_default_config(path: str = "config.yml") -> None:
    """Save default configuration to file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(create_default_config())


def save_default_scenario_config(path: str = "scenario.yml") -> None:
    """Save default scenario configuration to file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(create_default_scenario_config())


def init_default_configs(config_path: str = "config.yml", scenario_path: str = "scenario.yml") -> None:
    """Initialize both config and scenario files."""
    save_default_config(config_path)
    save_default_scenario_config(scenario_path)
