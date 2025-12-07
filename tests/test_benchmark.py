"""
Tests for LLM Benchmark Tool
"""

import pytest
import asyncio
import base64
import csv
import json
from pathlib import Path
import tempfile
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import (
    BenchmarkConfig,
    APIConfig,
    ModelConfig,
    MockDataConfig,
    ScenarioConfig,
    ScenarioDefaults,
    load_config,
    load_scenarios,
    parse_config,
    create_default_config,
    create_default_scenario_config,
    save_default_config,
    save_default_scenario_config,
    init_default_configs
)
from src.mock_data import (
    get_mock_generator,
    load_image_as_base64,
    MockRequest,
    BaseMockGenerator,
    ChatMockGenerator,
    EmbedMockGenerator,
    RerankerMockGenerator,
    VisionMockGenerator
)
from src.metrics import (
    BenchmarkMetrics,
    MetricsCollector,
    RequestMetrics
)
from src.exporters import (
    BaseExporter,
    MarkdownExporter,
    CSVExporter,
    JSONExporter,
    get_exporter,
    export_results
)


# ============================================================================
# Config Tests
# ============================================================================

class TestAPIConfig:
    """Tests for APIConfig dataclass."""
    
    def test_api_config_defaults(self):
        """Test APIConfig with defaults."""
        config = APIConfig(base_url="http://localhost:8000")
        assert config.base_url == "http://localhost:8000"
        assert config.api_key == ""
        assert config.timeout == 60
        assert config.endpoint_prefix == "/v1"
    
    def test_api_config_custom_values(self):
        """Test APIConfig with custom values."""
        config = APIConfig(
            base_url="http://api.example.com",
            api_key="secret-key",
            timeout=120
        )
        assert config.base_url == "http://api.example.com"
        assert config.api_key == "secret-key"
        assert config.timeout == 120
    
    def test_api_config_litellm_proxy(self):
        """Test APIConfig for LiteLLM proxy (no endpoint prefix)."""
        config = APIConfig(
            base_url="http://localhost:4000",
            endpoint_prefix=""
        )
        assert config.base_url == "http://localhost:4000"
        assert config.endpoint_prefix == ""


class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig with defaults."""
        config = ModelConfig(name="gpt-4", type="chat")
        assert config.name == "gpt-4"
        assert config.type == "chat"
        assert config.max_tokens == 32
        assert config.temperature == 0.2
    
    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            name="text-embedding-3-large",
            type="embed",
            max_tokens=100,
            temperature=0.0
        )
        assert config.name == "text-embedding-3-large"
        assert config.type == "embed"
        assert config.max_tokens == 100
        assert config.temperature == 0.0


class TestMockDataConfig:
    """Tests for MockDataConfig dataclass."""
    
    def test_mock_data_config_defaults(self):
        """Test MockDataConfig defaults."""
        config = MockDataConfig()
        assert len(config.chat_prompts) == 5
        assert len(config.embed_texts) == 5
        assert config.reranker_query == "What is machine learning?"
        assert len(config.reranker_documents) == 5
        assert len(config.vision_prompts) == 3
        assert config.vision_image_url is not None
        assert config.vision_image_base64 is None
        assert config.vision_image_path is None
        assert config.vision_image_paths == []
    
    def test_mock_data_config_custom_prompts(self):
        """Test MockDataConfig with custom prompts."""
        config = MockDataConfig(
            chat_prompts=["Test prompt 1", "Test prompt 2"],
            embed_texts=["Test text"]
        )
        assert len(config.chat_prompts) == 2
        assert len(config.embed_texts) == 1


class TestScenarioConfig:
    """Tests for ScenarioConfig dataclass."""
    
    def test_scenario_config_required_fields(self):
        """Test ScenarioConfig with required fields."""
        config = ScenarioConfig(name="test", requests=100, concurrency=10)
        assert config.name == "test"
        assert config.requests == 100
        assert config.concurrency == 10
        assert config.description == ""
        assert config.warmup_requests == 1
        assert config.timeout == 60
        assert config.enabled is True
        assert config.mode == "parallel"
    
    def test_scenario_config_all_fields(self):
        """Test ScenarioConfig with all fields."""
        config = ScenarioConfig(
            name="stress_test",
            requests=500,
            concurrency=50,
            description="High load test",
            warmup_requests=5,
            timeout=120,
            enabled=False,
            mode="queue_test"
        )
        assert config.name == "stress_test"
        assert config.requests == 500
        assert config.concurrency == 50
        assert config.description == "High load test"
        assert config.warmup_requests == 5
        assert config.timeout == 120
        assert config.enabled is False
        assert config.mode == "queue_test"


class TestScenarioDefaults:
    """Tests for ScenarioDefaults dataclass."""
    
    def test_scenario_defaults(self):
        """Test ScenarioDefaults default values."""
        defaults = ScenarioDefaults()
        assert defaults.requests == 100
        assert defaults.concurrency == 10
        assert defaults.mode == "parallel"
        assert defaults.warmup_requests == 1
        assert defaults.timeout == 60
        assert defaults.enabled is True


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""
    
    def test_benchmark_config_required_fields(self):
        """Test BenchmarkConfig with required fields."""
        config = BenchmarkConfig(
            api=APIConfig(base_url="http://localhost:8000"),
            model=ModelConfig(name="gpt-4", type="chat")
        )
        assert config.api.base_url == "http://localhost:8000"
        assert config.model.name == "gpt-4"
        assert config.scenarios == []
        assert config.default_requests == 100
        assert config.default_concurrency == 10
        assert config.capture_request_response is False
        assert config.output_dir == "results"
        assert config.export_formats == ["markdown", "csv"]
        assert config.quiet is False


class TestConfigFunctions:
    """Tests for config loading and saving functions."""
    
    def test_create_default_config(self):
        """Test default config creation."""
        config_str = create_default_config()
        assert "api:" in config_str
        assert "model:" in config_str
        assert "scenario_file:" in config_str
        assert "benchmark:" in config_str
        assert "mock_data:" in config_str
    
    def test_create_default_scenario_config(self):
        """Test default scenario config creation."""
        config_str = create_default_scenario_config()
        assert "scenarios:" in config_str
        assert "defaults:" in config_str
        assert "light_load" in config_str
        assert "warmup_requests:" in config_str
    
    def test_save_and_load_config(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yml"
            save_default_config(str(config_path))
            
            assert config_path.exists()
            
            config = load_config(str(config_path))
            assert config.api.base_url == "http://localhost:8000"
            assert config.model.type == "chat"
    
    def test_save_default_scenario_config(self):
        """Test saving scenario config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.yml"
            save_default_scenario_config(str(scenario_path))
            
            assert scenario_path.exists()
            content = scenario_path.read_text()
            assert "scenarios:" in content
    
    def test_init_default_configs(self):
        """Test initializing both config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = str(Path(tmpdir) / "config.yml")
            scenario_path = str(Path(tmpdir) / "scenario.yml")
            init_default_configs(config_path, scenario_path)
            
            assert Path(config_path).exists()
            assert Path(scenario_path).exists()
    
    def test_load_scenarios(self):
        """Test loading scenarios from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.yml"
            save_default_scenario_config(str(scenario_path))
            
            defaults = ScenarioDefaults()
            scenarios = load_scenarios(scenario_path, defaults)
            
            assert len(scenarios) > 0
            assert any(s.name == "light_load" for s in scenarios)
    
    def test_load_scenarios_nonexistent_file(self):
        """Test loading scenarios from nonexistent file."""
        defaults = ScenarioDefaults()
        scenarios = load_scenarios(Path("/nonexistent/path.yml"), defaults)
        assert scenarios == []
    
    def test_parse_config(self):
        """Test config parsing."""
        raw_config = {
            "api": {
                "base_url": "http://test:8000",
                "api_key": "test-key",
                "timeout": 30
            },
            "model": {
                "name": "test-model",
                "type": "embed",
                "max_tokens": 64
            },
            "scenarios": [
                {"name": "test", "requests": 10, "concurrency": 2}
            ],
            "benchmark": {
                "default_requests": 50,
                "capture_request_response": True
            }
        }
        
        config = parse_config(raw_config)
        assert config.api.base_url == "http://test:8000"
        assert config.api.api_key == "test-key"
        assert config.model.name == "test-model"
        assert config.model.type == "embed"
        assert config.capture_request_response is True
    
    def test_parse_config_with_vision_paths(self):
        """Test config parsing with vision image paths."""
        raw_config = {
            "api": {"base_url": "http://localhost:8000"},
            "model": {"name": "gpt-4-vision", "type": "vision"},
            "mock_data": {
                "vision_image_path": "/path/to/image.jpg",
                "vision_image_paths": ["/path/1.jpg", "/path/2.png"]
            }
        }
        
        config = parse_config(raw_config)
        assert config.mock_data.vision_image_path == "/path/to/image.jpg"
        assert len(config.mock_data.vision_image_paths) == 2
    
    def test_load_config_file_not_found(self):
        """Test loading nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yml")


# ============================================================================
# Mock Data Tests
# ============================================================================

class TestMockRequest:
    """Tests for MockRequest dataclass."""
    
    def test_mock_request(self):
        """Test MockRequest creation."""
        request = MockRequest(
            payload={"model": "test"},
            endpoint="/chat/completions",
            description="Test request"
        )
        assert request.payload == {"model": "test"}
        assert request.endpoint == "/chat/completions"
        assert request.description == "Test request"


class TestLoadImageAsBase64:
    """Tests for load_image_as_base64 function."""
    
    def test_load_image_as_base64_jpeg(self):
        """Test loading JPEG image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal JPEG-like file
            image_path = Path(tmpdir) / "test.jpg"
            image_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            image_path.write_bytes(image_data)
            
            base64_data, mime_type = load_image_as_base64(str(image_path))
            
            assert base64_data == base64.b64encode(image_data).decode("utf-8")
            assert mime_type == "image/jpeg"
    
    def test_load_image_as_base64_png(self):
        """Test loading PNG image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "test.png"
            image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
            image_path.write_bytes(image_data)
            
            base64_data, mime_type = load_image_as_base64(str(image_path))
            
            assert base64_data == base64.b64encode(image_data).decode("utf-8")
            assert mime_type == "image/png"
    
    def test_load_image_as_base64_file_not_found(self):
        """Test loading nonexistent image."""
        with pytest.raises(FileNotFoundError):
            load_image_as_base64("/nonexistent/image.jpg")


class TestChatMockGenerator:
    """Tests for ChatMockGenerator."""
    
    def test_get_endpoint(self):
        """Test endpoint."""
        model_config = ModelConfig(name="test", type="chat")
        mock_config = MockDataConfig()
        generator = ChatMockGenerator(model_config, mock_config)
        
        assert generator.get_endpoint() == "/chat/completions"
    
    def test_generate_single(self):
        """Test generating single request."""
        model_config = ModelConfig(name="gpt-4", type="chat", max_tokens=50)
        mock_config = MockDataConfig(chat_prompts=["Hello!"])
        generator = ChatMockGenerator(model_config, mock_config)
        
        requests = generator.generate(1)
        
        assert len(requests) == 1
        assert requests[0].payload["model"] == "gpt-4"
        assert requests[0].payload["max_tokens"] == 50
        assert requests[0].payload["messages"][0]["content"] == "Hello!"
    
    def test_generate_multiple(self):
        """Test generating multiple requests."""
        model_config = ModelConfig(name="test", type="chat")
        mock_config = MockDataConfig()
        generator = ChatMockGenerator(model_config, mock_config)
        
        requests = generator.generate(10)
        
        assert len(requests) == 10
        for req in requests:
            assert req.endpoint == "/chat/completions"
            assert "messages" in req.payload
            assert req.payload["stream"] is False


class TestEmbedMockGenerator:
    """Tests for EmbedMockGenerator."""
    
    def test_get_endpoint(self):
        """Test endpoint."""
        model_config = ModelConfig(name="test", type="embed")
        mock_config = MockDataConfig()
        generator = EmbedMockGenerator(model_config, mock_config)
        
        assert generator.get_endpoint() == "/embeddings"
    
    def test_generate(self):
        """Test generating embedding requests."""
        model_config = ModelConfig(name="text-embedding-3-small", type="embed")
        mock_config = MockDataConfig(embed_texts=["Test text"])
        generator = EmbedMockGenerator(model_config, mock_config)
        
        requests = generator.generate(3)
        
        assert len(requests) == 3
        for req in requests:
            assert req.endpoint == "/embeddings"
            assert req.payload["model"] == "text-embedding-3-small"
            assert req.payload["input"] == "Test text"
            assert req.payload["encoding_format"] == "float"


class TestRerankerMockGenerator:
    """Tests for RerankerMockGenerator."""
    
    def test_get_endpoint(self):
        """Test endpoint."""
        model_config = ModelConfig(name="test", type="reranker")
        mock_config = MockDataConfig()
        generator = RerankerMockGenerator(model_config, mock_config)
        
        assert generator.get_endpoint() == "/rerank"
    
    def test_generate(self):
        """Test generating reranker requests."""
        model_config = ModelConfig(name="rerank-model", type="reranker")
        mock_config = MockDataConfig(
            reranker_query="test query",
            reranker_documents=["doc1", "doc2", "doc3"]
        )
        generator = RerankerMockGenerator(model_config, mock_config)
        
        requests = generator.generate(2)
        
        assert len(requests) == 2
        for req in requests:
            assert req.endpoint == "/rerank"
            assert req.payload["query"] == "test query"
            assert len(req.payload["documents"]) == 3
            assert req.payload["top_n"] == 3


class TestVisionMockGenerator:
    """Tests for VisionMockGenerator."""
    
    def test_get_endpoint(self):
        """Test endpoint."""
        model_config = ModelConfig(name="test", type="vision")
        mock_config = MockDataConfig()
        generator = VisionMockGenerator(model_config, mock_config)
        
        assert generator.get_endpoint() == "/chat/completions"
    
    def test_generate_with_url(self):
        """Test generating with image URL."""
        model_config = ModelConfig(name="gpt-4-vision", type="vision")
        mock_config = MockDataConfig(
            vision_prompts=["Describe this"],
            vision_image_url="http://example.com/image.jpg"
        )
        generator = VisionMockGenerator(model_config, mock_config)
        
        requests = generator.generate(1)
        
        assert len(requests) == 1
        content = requests[0].payload["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "http://example.com/image.jpg"
    
    def test_generate_with_base64(self):
        """Test generating with base64 image."""
        model_config = ModelConfig(name="gpt-4-vision", type="vision")
        mock_config = MockDataConfig(
            vision_prompts=["Describe this"],
            vision_image_base64="dGVzdA=="
        )
        generator = VisionMockGenerator(model_config, mock_config)
        
        requests = generator.generate(1)
        
        content = requests[0].payload["messages"][0]["content"]
        assert "data:image/jpeg;base64,dGVzdA==" in content[1]["image_url"]["url"]
    
    def test_generate_with_local_path(self):
        """Test generating with local image path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "test.jpg"
            image_path.write_bytes(b"fake image data")
            
            model_config = ModelConfig(name="gpt-4-vision", type="vision")
            mock_config = MockDataConfig(
                vision_prompts=["Describe this"],
                vision_image_path=str(image_path)
            )
            generator = VisionMockGenerator(model_config, mock_config)
            
            requests = generator.generate(1)
            
            content = requests[0].payload["messages"][0]["content"]
            assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    
    def test_generate_with_multiple_paths(self):
        """Test generating with multiple local image paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                image_path = Path(tmpdir) / f"test{i}.jpg"
                image_path.write_bytes(f"fake image data {i}".encode())
                paths.append(str(image_path))
            
            model_config = ModelConfig(name="gpt-4-vision", type="vision")
            mock_config = MockDataConfig(
                vision_prompts=["Describe this"],
                vision_image_paths=paths
            )
            generator = VisionMockGenerator(model_config, mock_config)
            
            requests = generator.generate(5)
            
            assert len(requests) == 5
            # Check that images cycle through
            for i, req in enumerate(requests):
                content = req.payload["messages"][0]["content"]
                assert content[1]["type"] == "image_url"


class TestGetMockGenerator:
    """Tests for get_mock_generator factory function."""
    
    def test_get_chat_generator(self):
        """Test getting chat generator."""
        model_config = ModelConfig(name="test", type="chat")
        mock_config = MockDataConfig()
        
        generator = get_mock_generator(model_config, mock_config)
        assert isinstance(generator, ChatMockGenerator)
    
    def test_get_embed_generator(self):
        """Test getting embed generator."""
        model_config = ModelConfig(name="test", type="embed")
        mock_config = MockDataConfig()
        
        generator = get_mock_generator(model_config, mock_config)
        assert isinstance(generator, EmbedMockGenerator)
    
    def test_get_embedding_generator(self):
        """Test getting embedding generator (alias)."""
        model_config = ModelConfig(name="test", type="embedding")
        mock_config = MockDataConfig()
        
        generator = get_mock_generator(model_config, mock_config)
        assert isinstance(generator, EmbedMockGenerator)
    
    def test_get_reranker_generator(self):
        """Test getting reranker generator."""
        model_config = ModelConfig(name="test", type="reranker")
        mock_config = MockDataConfig()
        
        generator = get_mock_generator(model_config, mock_config)
        assert isinstance(generator, RerankerMockGenerator)
    
    def test_get_rerank_generator(self):
        """Test getting rerank generator (alias)."""
        model_config = ModelConfig(name="test", type="rerank")
        mock_config = MockDataConfig()
        
        generator = get_mock_generator(model_config, mock_config)
        assert isinstance(generator, RerankerMockGenerator)
    
    def test_get_vision_generator(self):
        """Test getting vision generator."""
        model_config = ModelConfig(name="test", type="vision")
        mock_config = MockDataConfig()
        
        generator = get_mock_generator(model_config, mock_config)
        assert isinstance(generator, VisionMockGenerator)
    
    def test_get_unknown_generator(self):
        """Test getting unknown generator type."""
        model_config = ModelConfig(name="test", type="unknown")
        mock_config = MockDataConfig()
        
        with pytest.raises(ValueError, match="Unknown model type"):
            get_mock_generator(model_config, mock_config)


# ============================================================================
# Metrics Tests
# ============================================================================

class TestRequestMetrics:
    """Tests for RequestMetrics dataclass."""
    
    def test_request_metrics_defaults(self):
        """Test RequestMetrics with defaults."""
        metrics = RequestMetrics(latency=0.5)
        assert metrics.latency == 0.5
        assert metrics.tokens == 0
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.response is None
    
    def test_request_metrics_all_fields(self):
        """Test RequestMetrics with all fields."""
        response = {"choices": []}
        metrics = RequestMetrics(
            latency=0.5,
            tokens=100,
            prompt_tokens=30,
            completion_tokens=70,
            success=False,
            error="Timeout",
            response=response
        )
        assert metrics.latency == 0.5
        assert metrics.tokens == 100
        assert metrics.prompt_tokens == 30
        assert metrics.completion_tokens == 70
        assert metrics.success is False
        assert metrics.error == "Timeout"
        assert metrics.response == response


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics dataclass."""
    
    def test_benchmark_metrics_defaults(self):
        """Test BenchmarkMetrics with defaults."""
        metrics = BenchmarkMetrics(
            model_name="test",
            model_type="chat",
            scenario_name="default",
            total_requests=100,
            concurrency=10
        )
        assert metrics.model_name == "test"
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.duration == 0.0
    
    def test_benchmark_metrics_calculate(self):
        """Test metrics calculation."""
        metrics = BenchmarkMetrics(
            model_name="test",
            model_type="chat",
            scenario_name="test_scenario",
            total_requests=5,
            concurrency=2
        )
        
        metrics.request_metrics = [
            RequestMetrics(latency=0.1, tokens=10, prompt_tokens=3, completion_tokens=7),
            RequestMetrics(latency=0.2, tokens=15, prompt_tokens=5, completion_tokens=10),
            RequestMetrics(latency=0.15, tokens=12, prompt_tokens=4, completion_tokens=8),
            RequestMetrics(latency=0.3, tokens=20, prompt_tokens=8, completion_tokens=12),
            RequestMetrics(latency=0.25, tokens=18, prompt_tokens=6, completion_tokens=12),
        ]
        metrics.duration = 1.0
        
        metrics.calculate()
        
        assert metrics.successful_requests == 5
        assert metrics.failed_requests == 0
        assert metrics.total_tokens == 75
        assert metrics.total_prompt_tokens == 26
        assert metrics.total_completion_tokens == 49
        assert metrics.requests_per_sec == 5.0
        assert metrics.tokens_per_sec == 75.0
        assert metrics.avg_latency == 0.2
        assert metrics.avg_tokens_per_request == 15.0
    
    def test_benchmark_metrics_calculate_with_failures(self):
        """Test metrics calculation with failed requests."""
        metrics = BenchmarkMetrics(
            model_name="test",
            model_type="chat",
            scenario_name="test",
            total_requests=3,
            concurrency=2
        )
        
        metrics.request_metrics = [
            RequestMetrics(latency=0.1, tokens=10, success=True),
            RequestMetrics(latency=0.2, tokens=0, success=False, error="Error"),
            RequestMetrics(latency=0.15, tokens=15, success=True),
        ]
        metrics.duration = 0.5
        
        metrics.calculate()
        
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.total_tokens == 25
    
    def test_benchmark_metrics_calculate_empty(self):
        """Test metrics calculation with no requests."""
        metrics = BenchmarkMetrics(
            model_name="test",
            model_type="chat",
            scenario_name="test",
            total_requests=0,
            concurrency=1
        )
        
        metrics.calculate()
        
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
    
    def test_benchmark_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = BenchmarkMetrics(
            model_name="test",
            model_type="chat",
            scenario_name="test",
            total_requests=10,
            concurrency=5
        )
        metrics.request_metrics = [
            RequestMetrics(latency=0.1, tokens=10)
        ]
        metrics.duration = 0.5
        metrics.calculate()
        
        data = metrics.to_dict()
        
        assert data["model_name"] == "test"
        assert data["model_type"] == "chat"
        assert data["scenario_name"] == "test"
        assert "configuration" in data
        assert "results" in data
        assert "latency" in data
        assert "tokens" in data
    
    def test_benchmark_metrics_latency_percentiles(self):
        """Test latency percentile calculations."""
        metrics = BenchmarkMetrics(
            model_name="test",
            model_type="chat",
            scenario_name="test",
            total_requests=100,
            concurrency=10
        )
        
        # Create 100 requests with latencies 0.01 to 1.0
        metrics.request_metrics = [
            RequestMetrics(latency=i/100, tokens=10)
            for i in range(1, 101)
        ]
        metrics.duration = 1.0
        
        metrics.calculate()
        
        assert metrics.min_latency == 0.01
        assert metrics.max_latency == 1.0
        assert 0.49 <= metrics.p50_latency <= 0.51
        assert 0.94 <= metrics.p95_latency <= 0.96
        assert 0.98 <= metrics.p99_latency <= 1.0


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_create_benchmark(self):
        """Test creating benchmark."""
        collector = MetricsCollector()
        
        benchmark = collector.create_benchmark(
            model_name="test",
            model_type="chat",
            scenario_name="test",
            total_requests=100,
            concurrency=10
        )
        
        assert benchmark.model_name == "test"
        assert benchmark.total_requests == 100
        assert len(collector.benchmarks) == 1
    
    def test_add_request_metric(self):
        """Test adding request metric."""
        collector = MetricsCollector()
        
        benchmark = collector.create_benchmark(
            model_name="test",
            model_type="chat",
            scenario_name="test",
            total_requests=10,
            concurrency=5
        )
        
        collector.add_request_metric(
            benchmark,
            latency=0.1,
            tokens=10,
            prompt_tokens=3,
            completion_tokens=7,
            success=True
        )
        
        assert len(benchmark.request_metrics) == 1
        assert benchmark.request_metrics[0].latency == 0.1
        assert benchmark.request_metrics[0].tokens == 10
    
    def test_finalize_benchmark(self):
        """Test finalizing benchmark."""
        collector = MetricsCollector()
        
        benchmark = collector.create_benchmark(
            model_name="test",
            model_type="chat",
            scenario_name="test",
            total_requests=2,
            concurrency=1
        )
        
        collector.add_request_metric(benchmark, latency=0.1, tokens=10, success=True)
        collector.add_request_metric(benchmark, latency=0.2, tokens=20, success=True)
        
        collector.finalize_benchmark(benchmark, duration=0.5)
        
        assert benchmark.duration == 0.5
        assert benchmark.end_time is not None
        assert benchmark.successful_requests == 2
    
    def test_get_summary(self):
        """Test getting summary."""
        collector = MetricsCollector()
        
        benchmark = collector.create_benchmark(
            model_name="test",
            model_type="chat",
            scenario_name="test",
            total_requests=1,
            concurrency=1
        )
        collector.add_request_metric(benchmark, latency=0.1, tokens=10, success=True)
        collector.finalize_benchmark(benchmark, duration=0.1)
        
        summary = collector.get_summary()
        
        assert summary["total_benchmarks"] == 1
        assert len(summary["benchmarks"]) == 1


# ============================================================================
# Exporter Tests
# ============================================================================

class TestBaseExporter:
    """Tests for BaseExporter."""
    
    def test_generate_filename(self):
        """Test filename generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = MarkdownExporter(tmpdir)
            filename = exporter.generate_filename("test")
            
            assert filename.startswith("test_")
            assert filename.endswith(".md")
    
    def test_output_dir_creation(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"
            exporter = MarkdownExporter(str(output_dir))
            
            assert output_dir.exists()


class TestMarkdownExporter:
    """Tests for MarkdownExporter."""
    
    def test_get_extension(self):
        """Test extension."""
        exporter = MarkdownExporter(".")
        assert exporter.get_extension() == "md"
    
    def test_export(self):
        """Test Markdown export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = BenchmarkMetrics(
                model_name="gpt-4",
                model_type="chat",
                scenario_name="test",
                total_requests=10,
                concurrency=5
            )
            metrics.request_metrics = [
                RequestMetrics(latency=0.1, tokens=10)
            ]
            metrics.duration = 0.5
            metrics.calculate()
            
            exporter = MarkdownExporter(tmpdir)
            filepath = exporter.export([metrics])
            
            assert Path(filepath).exists()
            content = Path(filepath).read_text(encoding='utf-8')
            assert "# LLM Benchmark Results" in content
            assert "gpt-4" in content
            assert "test" in content
    
    def test_export_multiple_metrics(self):
        """Test exporting multiple benchmark results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics1 = BenchmarkMetrics(
                model_name="model1",
                model_type="chat",
                scenario_name="scenario1",
                total_requests=10,
                concurrency=5
            )
            metrics1.request_metrics = [RequestMetrics(latency=0.1, tokens=10)]
            metrics1.duration = 0.5
            metrics1.calculate()
            
            metrics2 = BenchmarkMetrics(
                model_name="model2",
                model_type="embed",
                scenario_name="scenario2",
                total_requests=20,
                concurrency=10
            )
            metrics2.request_metrics = [RequestMetrics(latency=0.2, tokens=20)]
            metrics2.duration = 1.0
            metrics2.calculate()
            
            exporter = MarkdownExporter(tmpdir)
            filepath = exporter.export([metrics1, metrics2])
            
            content = Path(filepath).read_text(encoding='utf-8')
            assert "model1" in content
            assert "model2" in content
            assert "scenario1" in content
            assert "scenario2" in content


class TestCSVExporter:
    """Tests for CSVExporter."""
    
    def test_get_extension(self):
        """Test extension."""
        exporter = CSVExporter(".")
        assert exporter.get_extension() == "csv"
    
    def test_export(self):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = BenchmarkMetrics(
                model_name="test",
                model_type="chat",
                scenario_name="test",
                total_requests=10,
                concurrency=5
            )
            metrics.request_metrics = [
                RequestMetrics(latency=0.1, tokens=10)
            ]
            metrics.duration = 0.5
            metrics.calculate()
            
            exporter = CSVExporter(tmpdir)
            filepath = exporter.export([metrics])
            
            assert Path(filepath).exists()
            
            # Verify CSV content
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                row = next(reader)
                
                assert "scenario_name" in headers
                assert "model_name" in headers
                assert "test" in row


class TestJSONExporter:
    """Tests for JSONExporter."""
    
    def test_get_extension(self):
        """Test extension."""
        exporter = JSONExporter(".")
        assert exporter.get_extension() == "json"
    
    def test_export(self):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = BenchmarkMetrics(
                model_name="test",
                model_type="chat",
                scenario_name="test",
                total_requests=10,
                concurrency=5
            )
            metrics.request_metrics = [
                RequestMetrics(latency=0.1, tokens=10)
            ]
            metrics.duration = 0.5
            metrics.calculate()
            
            exporter = JSONExporter(tmpdir)
            filepath = exporter.export([metrics])
            
            assert Path(filepath).exists()
            
            # Verify JSON content
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                assert "generated_at" in data
                assert "benchmarks" in data
                assert len(data["benchmarks"]) == 1


class TestGetExporter:
    """Tests for get_exporter factory function."""
    
    def test_get_markdown_exporter(self):
        """Test getting markdown exporter."""
        exporter = get_exporter("markdown", ".")
        assert isinstance(exporter, MarkdownExporter)
    
    def test_get_md_exporter(self):
        """Test getting md exporter (alias)."""
        exporter = get_exporter("md", ".")
        assert isinstance(exporter, MarkdownExporter)
    
    def test_get_csv_exporter(self):
        """Test getting CSV exporter."""
        exporter = get_exporter("csv", ".")
        assert isinstance(exporter, CSVExporter)
    
    def test_get_json_exporter(self):
        """Test getting JSON exporter."""
        exporter = get_exporter("json", ".")
        assert isinstance(exporter, JSONExporter)
    
    def test_get_unknown_exporter(self):
        """Test getting unknown exporter type."""
        with pytest.raises(ValueError, match="Unknown export format"):
            get_exporter("unknown", ".")


class TestExportResults:
    """Tests for export_results function."""
    
    def test_export_results_multiple_formats(self):
        """Test exporting to multiple formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = BenchmarkMetrics(
                model_name="test",
                model_type="chat",
                scenario_name="test",
                total_requests=10,
                concurrency=5
            )
            metrics.request_metrics = [
                RequestMetrics(latency=0.1, tokens=10)
            ]
            metrics.duration = 0.5
            metrics.calculate()
            
            results = export_results([metrics], ["markdown", "csv", "json"], tmpdir)
            
            assert len(results) == 3
            assert "markdown" in results
            assert "csv" in results
            assert "json" in results
            
            for path in results.values():
                assert Path(path).exists()


# ==============================================================================
# Timeseries Tests
# ==============================================================================

from src.timeseries import (
    TimeseriesRecord,
    TimeseriesWriter,
    TimeseriesReader,
    load_all_timeseries
)


class TestTimeseriesRecord:
    """Tests for TimeseriesRecord dataclass."""
    
    def test_timeseries_record_creation(self):
        """Test creating a timeseries record."""
        record = TimeseriesRecord(
            timestamp=1700000000.0,
            elapsed_ms=1500.0,
            scenario_name="test_scenario",
            model_name="gpt-4",
            model_type="chat",
            request_id=1,
            latency_ms=150.5,
            success=True,
            tokens=100
        )
        assert record.timestamp == 1700000000.0
        assert record.elapsed_ms == 1500.0
        assert record.scenario_name == "test_scenario"
        assert record.model_name == "gpt-4"
        assert record.latency_ms == 150.5
        assert record.success is True
        assert record.tokens == 100
    
    def test_timeseries_record_to_dict(self):
        """Test converting record to dict."""
        record = TimeseriesRecord(
            timestamp=1700000000.0,
            elapsed_ms=1500.0,
            scenario_name="test",
            model_name="gpt-4",
            model_type="chat",
            request_id=1,
            latency_ms=100.0,
            success=True
        )
        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["timestamp"] == 1700000000.0
        assert d["scenario_name"] == "test"


class TestTimeseriesWriter:
    """Tests for TimeseriesWriter."""
    
    def test_writer_csv_creation(self):
        """Test creating CSV timeseries file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TimeseriesWriter(output_dir=tmpdir, format="csv")
            filepath = writer.start_scenario("test_scenario", "gpt-4")
            
            assert filepath.endswith(".csv")
            assert "test_scenario" in filepath
            assert "gpt-4" in filepath
            
            # End scenario to close file handle
            writer.end_scenario()
    
    def test_writer_record_data(self):
        """Test recording data to timeseries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TimeseriesWriter(output_dir=tmpdir, format="csv")
            writer.start_scenario("test", "model1")
            
            writer.record(
                scenario_name="test",
                model_name="model1",
                model_type="chat",
                latency=0.15,
                success=True,
                tokens=50
            )
            
            writer.record(
                scenario_name="test",
                model_name="model1",
                model_type="chat",
                latency=0.20,
                success=False,
                error="Timeout"
            )
            
            records = writer.end_scenario()
            assert len(records) == 2
            assert records[0].latency_ms == 150.0  # Converted to ms
            assert records[0].success is True
            assert records[1].success is False
            assert records[1].error == "Timeout"
    
    def test_writer_jsonl_format(self):
        """Test JSONL format output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TimeseriesWriter(output_dir=tmpdir, format="jsonl")
            filepath = writer.start_scenario("test", "model")
            
            writer.record(
                scenario_name="test",
                model_name="model",
                model_type="chat",
                latency=0.1,
                success=True
            )
            
            writer.end_scenario()
            
            assert filepath.endswith(".jsonl")
            assert Path(filepath).exists()
            
            # Verify content
            with open(filepath, 'r') as f:
                import json
                line = f.readline()
                data = json.loads(line)
                assert data["scenario_name"] == "test"


class TestTimeseriesReader:
    """Tests for TimeseriesReader."""
    
    def _create_test_csv(self, tmpdir: str) -> str:
        """Create a test CSV file."""
        filepath = Path(tmpdir) / "timeseries_test_model_20231115.csv"
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=TimeseriesWriter.CSV_HEADERS)
            writer.writeheader()
            
            base_time = 1700000000.0
            for i in range(10):
                writer.writerow({
                    "timestamp": base_time + i * 0.1,
                    "elapsed_ms": i * 100,
                    "scenario_name": "test",
                    "model_name": "model",
                    "model_type": "chat",
                    "request_id": i + 1,
                    "latency_ms": 100 + i * 10,
                    "success": "True" if i < 8 else "False",
                    "status_code": "",
                    "tokens": 50 + i * 5,
                    "prompt_tokens": 30,
                    "completion_tokens": 20 + i * 5,
                    "error": "" if i < 8 else "Error occurred",
                    "concurrent_requests": 5
                })
        return str(filepath)
    
    def test_reader_load_csv(self):
        """Test loading CSV timeseries file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._create_test_csv(tmpdir)
            reader = TimeseriesReader(filepath)
            
            assert len(reader.records) == 10
            assert reader.records[0].scenario_name == "test"
    
    def test_reader_get_statistics(self):
        """Test calculating statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._create_test_csv(tmpdir)
            reader = TimeseriesReader(filepath)
            stats = reader.get_statistics()
            
            assert stats["total_requests"] == 10
            assert stats["successful_requests"] == 8
            assert stats["failed_requests"] == 2
            assert stats["success_rate"] == 80.0
            assert "latency_avg_ms" in stats
            assert "latency_p95_ms" in stats
    
    def test_reader_get_latency_over_time(self):
        """Test getting latency over time buckets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._create_test_csv(tmpdir)
            reader = TimeseriesReader(filepath)
            buckets = reader.get_latency_over_time(bucket_size_ms=500)
            
            assert len(buckets) > 0
            assert "elapsed_ms" in buckets[0]
            assert "avg_latency_ms" in buckets[0]
    
    def test_reader_get_latency_distribution(self):
        """Test getting latency distribution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self._create_test_csv(tmpdir)
            reader = TimeseriesReader(filepath)
            dist = reader.get_latency_distribution(bins=10)
            
            assert "bins" in dist
            assert "counts" in dist
            assert len(dist["bins"]) == 10
    
    def test_reader_file_not_found(self):
        """Test handling non-existent file."""
        with pytest.raises(FileNotFoundError):
            TimeseriesReader("/nonexistent/path.csv")


class TestLoadAllTimeseries:
    """Tests for load_all_timeseries function."""
    
    def test_load_multiple_files(self):
        """Test loading multiple timeseries files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            for name in ["scenario1", "scenario2"]:
                filepath = Path(tmpdir) / f"timeseries_{name}_model.csv"
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    import csv
                    writer = csv.DictWriter(f, fieldnames=TimeseriesWriter.CSV_HEADERS)
                    writer.writeheader()
                    writer.writerow({
                        "timestamp": 1700000000.0,
                        "elapsed_ms": 0,
                        "scenario_name": name,
                        "model_name": "model",
                        "model_type": "chat",
                        "request_id": 1,
                        "latency_ms": 100,
                        "success": "True",
                        "status_code": "",
                        "tokens": 50,
                        "prompt_tokens": 30,
                        "completion_tokens": 20,
                        "error": "",
                        "concurrent_requests": 1
                    })
            
            readers = load_all_timeseries(tmpdir)
            assert len(readers) == 2


# ==============================================================================
# HTML Report Tests
# ==============================================================================

from src.html_report import (
    HTMLReportGenerator,
    generate_html_report
)


class TestHTMLReportGenerator:
    """Tests for HTMLReportGenerator."""
    
    def _create_test_timeseries(self, tmpdir: str) -> str:
        """Create a test timeseries file."""
        filepath = Path(tmpdir) / "timeseries_test_model_20231115.csv"
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=TimeseriesWriter.CSV_HEADERS)
            writer.writeheader()
            
            base_time = 1700000000.0
            for i in range(20):
                writer.writerow({
                    "timestamp": base_time + i * 0.1,
                    "elapsed_ms": i * 100,
                    "scenario_name": "test_scenario",
                    "model_name": "gpt-4",
                    "model_type": "chat",
                    "request_id": i + 1,
                    "latency_ms": 100 + i * 5,
                    "success": "True" if i < 18 else "False",
                    "status_code": "",
                    "tokens": 50 + i * 2,
                    "prompt_tokens": 30,
                    "completion_tokens": 20 + i * 2,
                    "error": "" if i < 18 else "Rate limit",
                    "concurrent_requests": 5
                })
        return str(filepath)
    
    def test_generate_scenario_report(self):
        """Test generating a single scenario report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_file = self._create_test_timeseries(tmpdir)
            reader = TimeseriesReader(ts_file)
            
            report_dir = Path(tmpdir) / "reports"
            generator = HTMLReportGenerator(str(report_dir))
            report_path = generator.generate_scenario_report(reader)
            
            assert Path(report_path).exists()
            assert report_path.endswith(".html")
            
            # Verify content
            content = Path(report_path).read_text(encoding='utf-8')
            assert "LLM Benchmark Report" in content
            assert "test_scenario" in content
            assert "gpt-4" in content
            assert "Chart.js" in content  # Charts included
    
    def test_generate_index(self):
        """Test generating index page."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_file = self._create_test_timeseries(tmpdir)
            reader = TimeseriesReader(ts_file)
            
            report_dir = Path(tmpdir) / "reports"
            generator = HTMLReportGenerator(str(report_dir))
            generator.generate_scenario_report(reader)
            index_path = generator.generate_index([reader])
            
            assert Path(index_path).exists()
            content = Path(index_path).read_text(encoding='utf-8')
            assert "Summary" in content
            assert "test_scenario" in content
    
    def test_generate_from_directory(self):
        """Test generating reports from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_timeseries(tmpdir)
            
            report_dir = Path(tmpdir) / "reports"
            generator = HTMLReportGenerator(str(report_dir))
            index_path = generator.generate_from_directory(tmpdir)
            
            assert Path(index_path).exists()
            # Check individual report also created
            report_files = list(report_dir.glob("report_*.html"))
            assert len(report_files) >= 1


class TestGenerateHTMLReport:
    """Tests for generate_html_report convenience function."""
    
    def _create_test_timeseries(self, tmpdir: str) -> str:
        """Create a test timeseries file."""
        filepath = Path(tmpdir) / "timeseries_test_model_20231115.csv"
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=TimeseriesWriter.CSV_HEADERS)
            writer.writeheader()
            
            for i in range(10):
                writer.writerow({
                    "timestamp": 1700000000.0 + i * 0.1,
                    "elapsed_ms": i * 100,
                    "scenario_name": "test",
                    "model_name": "model",
                    "model_type": "chat",
                    "request_id": i + 1,
                    "latency_ms": 100,
                    "success": "True",
                    "status_code": "",
                    "tokens": 50,
                    "prompt_tokens": 30,
                    "completion_tokens": 20,
                    "error": "",
                    "concurrent_requests": 1
                })
        return str(filepath)
    
    def test_generate_from_file(self):
        """Test generating report from single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_file = self._create_test_timeseries(tmpdir)
            report_dir = Path(tmpdir) / "reports"
            
            index_path = generate_html_report(ts_file, str(report_dir))
            assert Path(index_path).exists()
    
    def test_generate_from_directory(self):
        """Test generating report from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_test_timeseries(tmpdir)
            report_dir = Path(tmpdir) / "reports"
            
            index_path = generate_html_report(tmpdir, str(report_dir))
            assert Path(index_path).exists()


# ============================================================================
# Engine JMeter-Style Mode Tests
# ============================================================================

from src.engine import BenchmarkMode, LoadProfile, QueueMetrics


class TestBenchmarkMode:
    """Tests for BenchmarkMode enum."""
    
    def test_basic_modes(self):
        """Test basic execution modes."""
        assert BenchmarkMode.PARALLEL.value == "parallel"
        assert BenchmarkMode.CONTROLLED.value == "controlled"
        assert BenchmarkMode.QUEUE_TEST.value == "queue_test"
    
    def test_jmeter_modes(self):
        """Test JMeter-style execution modes."""
        assert BenchmarkMode.RAMP_UP.value == "ramp_up"
        assert BenchmarkMode.STEPPING.value == "stepping"
        assert BenchmarkMode.SPIKE.value == "spike"
        assert BenchmarkMode.CONSTANT_RATE.value == "constant_rate"
        assert BenchmarkMode.ARRIVALS.value == "arrivals"
        assert BenchmarkMode.ULTIMATE.value == "ultimate"
        assert BenchmarkMode.DURATION.value == "duration"
    
    def test_mode_from_string(self):
        """Test creating mode from string value."""
        assert BenchmarkMode("parallel") == BenchmarkMode.PARALLEL
        assert BenchmarkMode("ramp_up") == BenchmarkMode.RAMP_UP
        assert BenchmarkMode("constant_rate") == BenchmarkMode.CONSTANT_RATE
        assert BenchmarkMode("ultimate") == BenchmarkMode.ULTIMATE


class TestLoadProfile:
    """Tests for LoadProfile dataclass."""
    
    def test_load_profile_defaults(self):
        """Test LoadProfile with default values."""
        profile = LoadProfile()
        assert profile.ramp_up_time == 0.0
        assert profile.ramp_up_steps == 1
        assert profile.hold_time == 0.0
        assert profile.ramp_down_time == 0.0
        assert profile.target_rps == 0.0
        assert profile.spike_multiplier == 2.0
        assert profile.spike_duration == 5.0
        assert profile.arrival_rate == 10.0
        assert profile.duration_seconds == 60.0
        assert profile.stages == []
    
    def test_load_profile_ramp_up(self):
        """Test LoadProfile for ramp-up mode."""
        profile = LoadProfile(
            ramp_up_time=30.0,
            ramp_up_steps=10,
            hold_time=60.0
        )
        assert profile.ramp_up_time == 30.0
        assert profile.ramp_up_steps == 10
        assert profile.hold_time == 60.0
    
    def test_load_profile_spike(self):
        """Test LoadProfile for spike testing."""
        profile = LoadProfile(
            spike_multiplier=5.0,
            spike_duration=10.0
        )
        assert profile.spike_multiplier == 5.0
        assert profile.spike_duration == 10.0
    
    def test_load_profile_constant_rate(self):
        """Test LoadProfile for constant rate mode."""
        profile = LoadProfile(
            target_rps=100.0
        )
        assert profile.target_rps == 100.0
    
    def test_load_profile_arrivals(self):
        """Test LoadProfile for arrivals mode."""
        profile = LoadProfile(
            arrival_rate=50.0
        )
        assert profile.arrival_rate == 50.0
    
    def test_load_profile_ultimate_stages(self):
        """Test LoadProfile with ultimate thread group stages."""
        stages = [
            (0, 10, 10.0),    # Ramp to 10 threads
            (10, 50, 20.0),   # Ramp to 50 threads
            (50, 50, 30.0),   # Hold at 50
            (50, 0, 10.0),    # Ramp down
        ]
        profile = LoadProfile(stages=stages)
        assert len(profile.stages) == 4
        assert profile.stages[0] == (0, 10, 10.0)
        assert profile.stages[2] == (50, 50, 30.0)
    
    def test_load_profile_duration(self):
        """Test LoadProfile for duration-based testing."""
        profile = LoadProfile(
            duration_seconds=120.0
        )
        assert profile.duration_seconds == 120.0


class TestQueueMetrics:
    """Tests for QueueMetrics dataclass."""
    
    def test_queue_metrics_defaults(self):
        """Test QueueMetrics with default values."""
        qm = QueueMetrics()
        assert qm.queue_depth_samples == []
        assert qm.wait_times == []
        assert qm.processing_times == []
        assert qm.rejection_count == 0
        assert qm.timeout_count == 0
        assert qm.max_observed_queue_depth == 0
        assert qm.avg_queue_depth == 0.0
    
    def test_queue_metrics_calculate(self):
        """Test QueueMetrics calculation."""
        qm = QueueMetrics()
        qm.queue_depth_samples = [
            (0.1, 5),
            (0.2, 10),
            (0.3, 8),
            (0.4, 3),
        ]
        qm.wait_times = [0.1, 0.2, 0.15]
        qm.processing_times = [0.5, 0.6, 0.55]
        qm.rejection_count = 2
        qm.timeout_count = 1
        
        qm.calculate()
        
        assert qm.max_observed_queue_depth == 10
        assert qm.avg_queue_depth == 6.5  # (5+10+8+3)/4
    
    def test_queue_metrics_calculate_empty(self):
        """Test QueueMetrics calculation with empty data."""
        qm = QueueMetrics()
        qm.calculate()
        assert qm.max_observed_queue_depth == 0
        assert qm.avg_queue_depth == 0.0


class TestScenarioConfigJMeterParams:
    """Tests for ScenarioConfig with JMeter-style parameters."""
    
    def test_scenario_config_jmeter_defaults(self):
        """Test ScenarioConfig JMeter parameters have defaults."""
        config = ScenarioConfig(
            name="test",
            requests=100,
            concurrency=10
        )
        assert config.mode == "parallel"
        assert config.ramp_up_time == 0.0
        assert config.ramp_up_steps == 1
        assert config.hold_time == 0.0
        assert config.target_rps == 0.0
        assert config.spike_multiplier == 2.0
        assert config.spike_duration == 5.0
        assert config.arrival_rate == 10.0
        assert config.duration_seconds == 60.0
        assert config.stages == []
    
    def test_scenario_config_ramp_up(self):
        """Test ScenarioConfig for ramp-up mode."""
        config = ScenarioConfig(
            name="ramp_test",
            requests=200,
            concurrency=40,
            mode="ramp_up",
            ramp_up_time=30.0,
            ramp_up_steps=10,
            hold_time=60.0
        )
        assert config.mode == "ramp_up"
        assert config.ramp_up_time == 30.0
        assert config.ramp_up_steps == 10
        assert config.hold_time == 60.0
    
    def test_scenario_config_spike(self):
        """Test ScenarioConfig for spike testing."""
        config = ScenarioConfig(
            name="spike_test",
            requests=200,
            concurrency=20,
            mode="spike",
            spike_multiplier=5.0,
            spike_duration=10.0
        )
        assert config.mode == "spike"
        assert config.spike_multiplier == 5.0
        assert config.spike_duration == 10.0
    
    def test_scenario_config_constant_rate(self):
        """Test ScenarioConfig for constant rate mode."""
        config = ScenarioConfig(
            name="constant_test",
            requests=300,
            concurrency=30,
            mode="constant_rate",
            target_rps=50.0
        )
        assert config.mode == "constant_rate"
        assert config.target_rps == 50.0
    
    def test_scenario_config_ultimate(self):
        """Test ScenarioConfig for ultimate thread group."""
        stages = [
            (0, 10, 10),
            (10, 50, 20),
            (50, 50, 30),
            (50, 0, 10),
        ]
        config = ScenarioConfig(
            name="ultimate_test",
            requests=500,
            concurrency=50,
            mode="ultimate",
            stages=stages
        )
        assert config.mode == "ultimate"
        assert len(config.stages) == 4


class TestLoadScenariosWithJMeterParams:
    """Tests for loading scenarios with JMeter-style parameters."""
    
    def test_load_scenarios_with_jmeter_params(self):
        """Test loading scenarios with all JMeter parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.yml"
            scenario_content = """
scenarios:
  - name: "ramp_test"
    requests: 200
    concurrency: 40
    mode: "ramp_up"
    ramp_up_time: 30.0
    ramp_up_steps: 10
    hold_time: 60.0
    
  - name: "spike_test"
    requests: 200
    concurrency: 20
    mode: "spike"
    spike_multiplier: 5.0
    spike_duration: 10.0
    
  - name: "constant_test"
    requests: 300
    concurrency: 30
    mode: "constant_rate"
    target_rps: 50.0
    
  - name: "arrivals_test"
    requests: 200
    concurrency: 100
    mode: "arrivals"
    arrival_rate: 25.0
    
  - name: "duration_test"
    requests: 100
    concurrency: 15
    mode: "duration"
    duration_seconds: 120.0
"""
            with open(scenario_path, 'w', encoding='utf-8') as f:
                f.write(scenario_content)
            
            scenarios = load_scenarios(scenario_path, ScenarioDefaults())
            
            assert len(scenarios) == 5
            
            # Check ramp-up scenario
            ramp = scenarios[0]
            assert ramp.name == "ramp_test"
            assert ramp.mode == "ramp_up"
            assert ramp.ramp_up_time == 30.0
            assert ramp.ramp_up_steps == 10
            assert ramp.hold_time == 60.0
            
            # Check spike scenario
            spike = scenarios[1]
            assert spike.mode == "spike"
            assert spike.spike_multiplier == 5.0
            assert spike.spike_duration == 10.0
            
            # Check constant rate scenario
            constant = scenarios[2]
            assert constant.mode == "constant_rate"
            assert constant.target_rps == 50.0
            
            # Check arrivals scenario
            arrivals = scenarios[3]
            assert arrivals.mode == "arrivals"
            assert arrivals.arrival_rate == 25.0
            
            # Check duration scenario
            duration = scenarios[4]
            assert duration.mode == "duration"
            assert duration.duration_seconds == 120.0
    
    def test_load_scenarios_with_stages(self):
        """Test loading scenarios with ultimate thread group stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.yml"
            scenario_content = """
scenarios:
  - name: "ultimate_test"
    requests: 500
    concurrency: 50
    mode: "ultimate"
    stages:
      - { start: 0, end: 10, duration: 10 }
      - { start: 10, end: 50, duration: 20 }
      - { start: 50, end: 50, duration: 30 }
      - { start: 50, end: 0, duration: 10 }
"""
            with open(scenario_path, 'w', encoding='utf-8') as f:
                f.write(scenario_content)
            
            scenarios = load_scenarios(scenario_path, ScenarioDefaults())
            
            assert len(scenarios) == 1
            ultimate = scenarios[0]
            assert ultimate.mode == "ultimate"
            assert len(ultimate.stages) == 4
            assert ultimate.stages[0] == (0, 10, 10)
            assert ultimate.stages[1] == (10, 50, 20)
            assert ultimate.stages[2] == (50, 50, 30)
            assert ultimate.stages[3] == (50, 0, 10)


# ============================================================================
# Debug Logger Tests
# ============================================================================

class TestDebugLogger:
    """Tests for DebugLogger class."""
    
    def test_debug_logger_creation(self):
        """Test DebugLogger instantiation."""
        from src.debug_logger import DebugLogger
        import logging
        
        logger = DebugLogger(
            name="test_logger",
            level=logging.DEBUG,
            console_output=False,
            file_output=False
        )
        
        assert logger.console_output == False
        assert logger.file_output == False
        assert logger._start_time is None
        assert logger._request_count == 0
        assert logger._mode is None
    
    def test_debug_logger_level(self):
        """Test debug logger level changes."""
        from src.debug_logger import DebugLogger
        import logging
        
        logger = DebugLogger(
            name="test_level_logger",
            level=logging.DEBUG,
            console_output=False,
            file_output=False
        )
        
        assert logger.is_debug_enabled() == True
        
        logger.set_level(logging.WARNING)
        assert logger.is_debug_enabled() == False
        
        logger.set_level(logging.DEBUG)
        assert logger.is_debug_enabled() == True
    
    def test_debug_logger_elapsed(self):
        """Test elapsed time formatting."""
        from src.debug_logger import DebugLogger
        import logging
        import time
        
        logger = DebugLogger(
            name="test_elapsed_logger",
            level=logging.DEBUG,
            console_output=False,
            file_output=False
        )
        
        # Before start_time is set
        elapsed_ctx = logger._with_elapsed()
        assert elapsed_ctx == {}  # No elapsed time when start_time is not set
        
        # After start_time is set
        logger._start_time = time.perf_counter()
        time.sleep(0.01)
        elapsed_ctx = logger._with_elapsed()
        assert "_start_time" in elapsed_ctx  # Contains start_time for processor
        assert isinstance(elapsed_ctx["_start_time"], float)
    
    def test_debug_logger_mode_start(self):
        """Test mode_start sets state correctly."""
        from src.debug_logger import DebugLogger
        import logging
        
        logger = DebugLogger(
            name="test_mode_logger",
            level=logging.DEBUG,
            console_output=False,
            file_output=False
        )
        
        logger.mode_start("ramp_up", "test_scenario", {"key": "value"})
        
        assert logger._mode == "ramp_up"
        assert logger._request_count == 0
        assert logger._start_time is not None
    
    def test_debug_logger_request_submit(self):
        """Test request_submit increments counter."""
        from src.debug_logger import DebugLogger
        import logging
        
        logger = DebugLogger(
            name="test_request_logger",
            level=logging.DEBUG,
            console_output=False,
            file_output=False
        )
        
        assert logger._request_count == 0
        
        logger.request_submit(1, 5)
        assert logger._request_count == 1
        
        logger.request_submit(2, 6)
        assert logger._request_count == 2


class TestDebugLoggerGlobalFunctions:
    """Tests for global debug logger functions."""
    
    def test_get_debug_logger(self):
        """Test get_debug_logger returns consistent instance."""
        from src.debug_logger import get_debug_logger, reset_debug_logger
        
        # Reset first to ensure clean state
        reset_debug_logger()
        
        logger1 = get_debug_logger()
        logger2 = get_debug_logger()
        
        assert logger1 is logger2
    
    def test_reset_debug_logger(self):
        """Test reset_debug_logger creates new instance."""
        from src.debug_logger import get_debug_logger, reset_debug_logger
        
        logger1 = get_debug_logger()
        logger2 = reset_debug_logger(console_output=True)
        
        assert logger2.console_output == True
    
    def test_enable_disable_debug_logging(self):
        """Test enable/disable debug logging."""
        from src.debug_logger import (
            enable_debug_logging, 
            disable_debug_logging, 
            get_debug_logger,
            reset_debug_logger
        )
        import logging
        
        reset_debug_logger()
        
        enable_debug_logging(logging.DEBUG)
        logger = get_debug_logger()
        assert logger.is_debug_enabled() == True
        
        disable_debug_logging()
        assert logger.is_debug_enabled() == False


# ============================================================================
# Mode Runner Tests
# ============================================================================

class TestModeRunners:
    """Tests for mode runner classes."""
    
    def test_get_mode_runner_ramp_up(self):
        """Test get_mode_runner returns RampUpRunner."""
        from src.mode_runners import get_mode_runner, RampUpRunner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.RAMP_UP)
        assert isinstance(runner, RampUpRunner)
    
    def test_get_mode_runner_stepping(self):
        """Test get_mode_runner returns SteppingRunner."""
        from src.mode_runners import get_mode_runner, SteppingRunner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.STEPPING)
        assert isinstance(runner, SteppingRunner)
    
    def test_get_mode_runner_spike(self):
        """Test get_mode_runner returns SpikeRunner."""
        from src.mode_runners import get_mode_runner, SpikeRunner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.SPIKE)
        assert isinstance(runner, SpikeRunner)
    
    def test_get_mode_runner_constant_rate(self):
        """Test get_mode_runner returns ConstantRateRunner."""
        from src.mode_runners import get_mode_runner, ConstantRateRunner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.CONSTANT_RATE)
        assert isinstance(runner, ConstantRateRunner)
    
    def test_get_mode_runner_arrivals(self):
        """Test get_mode_runner returns ArrivalsRunner."""
        from src.mode_runners import get_mode_runner, ArrivalsRunner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.ARRIVALS)
        assert isinstance(runner, ArrivalsRunner)
    
    def test_get_mode_runner_ultimate(self):
        """Test get_mode_runner returns UltimateRunner."""
        from src.mode_runners import get_mode_runner, UltimateRunner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.ULTIMATE)
        assert isinstance(runner, UltimateRunner)
    
    def test_get_mode_runner_duration(self):
        """Test get_mode_runner returns DurationRunner."""
        from src.mode_runners import get_mode_runner, DurationRunner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.DURATION)
        assert isinstance(runner, DurationRunner)
    
    def test_get_mode_runner_parallel_returns_none(self):
        """Test get_mode_runner returns None for PARALLEL mode."""
        from src.mode_runners import get_mode_runner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.PARALLEL)
        assert runner is None
    
    def test_get_mode_runner_controlled_returns_none(self):
        """Test get_mode_runner returns None for CONTROLLED mode."""
        from src.mode_runners import get_mode_runner
        from src.modes import BenchmarkMode
        
        runner = get_mode_runner(BenchmarkMode.CONTROLLED)
        assert runner is None
    
    def test_mode_runner_debug_flag(self):
        """Test mode runner accepts debug flag."""
        from src.mode_runners import RampUpRunner
        from src.debug_logger import DebugLogger
        import logging
        
        debug_logger = DebugLogger(
            name="test_runner_logger",
            level=logging.DEBUG,
            console_output=False,
            file_output=False
        )
        
        runner = RampUpRunner(debug=True, debug_logger=debug_logger)
        
        assert runner.debug == True
        assert runner._debug_logger is debug_logger


# ============================================================================
# Queue Metrics Extended Tests
# ============================================================================

class TestQueueMetricsExtended:
    """Extended tests for QueueMetrics class."""
    
    def test_queue_metrics_record_rejection(self):
        """Test recording rejections."""
        from src.modes import QueueMetrics
        
        qm = QueueMetrics()
        assert qm.rejection_count == 0
        
        qm.record_rejection()
        assert qm.rejection_count == 1
        
        qm.record_rejection()
        assert qm.rejection_count == 2
    
    def test_queue_metrics_record_timeout(self):
        """Test recording timeouts."""
        from src.modes import QueueMetrics
        
        qm = QueueMetrics()
        assert qm.timeout_count == 0
        
        qm.record_timeout()
        assert qm.timeout_count == 1
        
        qm.record_timeout()
        assert qm.timeout_count == 2
    
    def test_queue_metrics_add_queue_sample(self):
        """Test adding queue samples."""
        from src.modes import QueueMetrics
        
        qm = QueueMetrics()
        assert len(qm.queue_depth_samples) == 0
        
        qm.add_queue_sample(1.0, 5)
        assert len(qm.queue_depth_samples) == 1
        assert qm.queue_depth_samples[0] == (1.0, 5)
        
        qm.add_queue_sample(2.0, 10)
        assert len(qm.queue_depth_samples) == 2
    
    def test_queue_metrics_add_wait_time(self):
        """Test adding wait times with threshold."""
        from src.modes import QueueMetrics
        
        qm = QueueMetrics()
        
        # Small wait time should not be recorded
        qm.add_wait_time(0.0005)
        assert len(qm.wait_times) == 0
        
        # Larger wait time should be recorded
        qm.add_wait_time(0.01)
        assert len(qm.wait_times) == 1
        assert qm.wait_times[0] == 0.01
    
    def test_queue_metrics_add_processing_time(self):
        """Test adding processing times."""
        from src.modes import QueueMetrics
        
        qm = QueueMetrics()
        
        # None should not be recorded
        qm.add_processing_time(None)
        assert len(qm.processing_times) == 0
        
        # Zero should not be recorded
        qm.add_processing_time(0)
        assert len(qm.processing_times) == 0
        
        # Valid time should be recorded
        qm.add_processing_time(0.5)
        assert len(qm.processing_times) == 1
        assert qm.processing_times[0] == 0.5


# ============================================================================
# Load Profile Extended Tests
# ============================================================================

class TestLoadProfileExtended:
    """Extended tests for LoadProfile class."""
    
    def test_load_profile_from_scenario(self):
        """Test LoadProfile.from_scenario method."""
        from src.modes import LoadProfile
        from src.config import ScenarioConfig
        
        scenario = ScenarioConfig(
            name="test",
            requests=100,
            concurrency=10,
            ramp_up_time=30.0,
            ramp_up_steps=5,
            hold_time=60.0,
            target_rps=50.0,
            spike_multiplier=3.0,
            spike_duration=10.0,
            arrival_rate=20.0,
            duration_seconds=120.0
        )
        
        profile = LoadProfile.from_scenario(scenario)
        
        assert profile.ramp_up_time == 30.0
        assert profile.ramp_up_steps == 5
        assert profile.hold_time == 60.0
        assert profile.target_rps == 50.0
        assert profile.spike_multiplier == 3.0
        assert profile.spike_duration == 10.0
        assert profile.arrival_rate == 20.0
        assert profile.duration_seconds == 120.0
    
    def test_load_profile_from_scenario_with_stages(self):
        """Test LoadProfile.from_scenario with stages."""
        from src.modes import LoadProfile
        from src.config import ScenarioConfig
        
        scenario = ScenarioConfig(
            name="test",
            requests=100,
            concurrency=10,
            stages=[(0, 10, 5), (10, 20, 10), (20, 0, 5)]
        )
        
        profile = LoadProfile.from_scenario(scenario)
        
        assert len(profile.stages) == 3
        assert profile.stages[0] == (0, 10, 5)
        assert profile.stages[1] == (10, 20, 10)
        assert profile.stages[2] == (20, 0, 5)


# ============================================================================
# Benchmark Mode Extended Tests
# ============================================================================

class TestBenchmarkModeExtended:
    """Extended tests for BenchmarkMode enum."""
    
    def test_benchmark_mode_from_string_case_insensitive(self):
        """Test BenchmarkMode.from_string is case insensitive."""
        from src.modes import BenchmarkMode
        
        assert BenchmarkMode.from_string("PARALLEL") == BenchmarkMode.PARALLEL
        assert BenchmarkMode.from_string("Parallel") == BenchmarkMode.PARALLEL
        assert BenchmarkMode.from_string("parallel") == BenchmarkMode.PARALLEL
        
        assert BenchmarkMode.from_string("RAMP_UP") == BenchmarkMode.RAMP_UP
        assert BenchmarkMode.from_string("ramp_up") == BenchmarkMode.RAMP_UP
    
    def test_benchmark_mode_from_string_invalid(self):
        """Test BenchmarkMode.from_string with invalid value."""
        from src.modes import BenchmarkMode
        
        with pytest.raises(ValueError):
            BenchmarkMode.from_string("invalid_mode")
    
    def test_benchmark_mode_all_values(self):
        """Test all BenchmarkMode values can be converted."""
        from src.modes import BenchmarkMode
        
        for mode in BenchmarkMode:
            result = BenchmarkMode.from_string(mode.value)
            assert result == mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
