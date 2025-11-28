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
class ScenarioConfig:
    """Benchmark scenario configuration."""
    name: str
    requests: int
    concurrency: int
    description: str = ""
    warmup_requests: int = 1
    timeout: int = 60
    enabled: bool = True


@dataclass
class ScenarioDefaults:
    """Default settings for scenarios."""
    requests: int = 100
    concurrency: int = 10
    warmup_requests: int = 1
    timeout: int = 60
    enabled: bool = True


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    api: APIConfig
    model: ModelConfig
    mock_data: MockDataConfig = field(default_factory=MockDataConfig)
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
        enabled=file_defaults.get('enabled', defaults.enabled)
    )
    
    scenarios = []
    for s in scenario_data.get('scenarios', []):
        scenario = ScenarioConfig(
            name=s.get('name', 'default'),
            requests=s.get('requests', defaults.requests),
            concurrency=s.get('concurrency', defaults.concurrency),
            description=s.get('description', ''),
            warmup_requests=s.get('warmup_requests', defaults.warmup_requests),
            timeout=s.get('timeout', defaults.timeout),
            enabled=s.get('enabled', defaults.enabled)
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
                enabled=s.get('enabled', scenario_defaults.enabled)
            ))
    
    # Parse benchmark settings
    benchmark_raw = raw_config.get('benchmark', {})
    
    return BenchmarkConfig(
        api=api_config,
        model=model_config,
        mock_data=mock_config,
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

scenarios:
  - name: "light_load"
    description: "Light load test for basic validation"
    requests: 50
    concurrency: 5
    warmup_requests: 2
    timeout: 60
    enabled: true

  - name: "medium_load"
    description: "Medium load test for standard benchmarking"
    requests: 100
    concurrency: 10
    warmup_requests: 3
    timeout: 60
    enabled: true

  - name: "heavy_load"
    description: "Heavy load test for performance limits"
    requests: 200
    concurrency: 20
    warmup_requests: 5
    timeout: 90
    enabled: true

  - name: "stress_test"
    description: "Stress test with high concurrency"
    requests: 500
    concurrency: 50
    warmup_requests: 5
    timeout: 120
    enabled: true

# Default scenario settings (applied if not specified per scenario)
defaults:
  requests: 100
  concurrency: 10
  warmup_requests: 1
  timeout: 60
  enabled: true
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
