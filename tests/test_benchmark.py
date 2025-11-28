"""
Tests for LLM Benchmark Tool
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import os

from llm_benchmark.config import (
    BenchmarkConfig,
    APIConfig,
    ModelConfig,
    MockDataConfig,
    ScenarioConfig,
    load_config,
    parse_config,
    create_default_config,
    save_default_config
)
from llm_benchmark.mock_data import (
    get_mock_generator,
    ChatMockGenerator,
    EmbedMockGenerator,
    RerankerMockGenerator,
    VisionMockGenerator
)
from llm_benchmark.metrics import (
    BenchmarkMetrics,
    MetricsCollector,
    RequestMetrics
)
from llm_benchmark.exporters import (
    MarkdownExporter,
    CSVExporter,
    JSONExporter,
    export_results
)


class TestConfig:
    """Tests for configuration loading."""
    
    def test_create_default_config(self):
        """Test default config creation."""
        config_str = create_default_config()
        assert "api:" in config_str
        assert "model:" in config_str
        assert "scenarios:" in config_str
    
    def test_save_and_load_config(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yml"
            save_default_config(str(config_path))
            
            assert config_path.exists()
            
            config = load_config(str(config_path))
            assert config.api.base_url == "http://localhost:8000"
            assert config.model.type == "chat"
    
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
                "capture_responses": True
            }
        }
        
        config = parse_config(raw_config)
        assert config.api.base_url == "http://test:8000"
        assert config.api.api_key == "test-key"
        assert config.model.name == "test-model"
        assert config.model.type == "embed"
        assert len(config.scenarios) == 1
        assert config.capture_responses is True


class TestMockData:
    """Tests for mock data generators."""
    
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
    
    def test_chat_generator_generate(self):
        """Test chat generator produces valid requests."""
        model_config = ModelConfig(name="test-model", type="chat", max_tokens=32)
        mock_config = MockDataConfig()
        
        generator = ChatMockGenerator(model_config, mock_config)
        requests = generator.generate(5)
        
        assert len(requests) == 5
        for req in requests:
            assert req.endpoint == "/v1/chat/completions"
            assert req.payload["model"] == "test-model"
            assert "messages" in req.payload
    
    def test_embed_generator_generate(self):
        """Test embed generator produces valid requests."""
        model_config = ModelConfig(name="embed-model", type="embed")
        mock_config = MockDataConfig()
        
        generator = EmbedMockGenerator(model_config, mock_config)
        requests = generator.generate(3)
        
        assert len(requests) == 3
        for req in requests:
            assert req.endpoint == "/v1/embeddings"
            assert req.payload["model"] == "embed-model"
            assert "input" in req.payload
    
    def test_reranker_generator_generate(self):
        """Test reranker generator produces valid requests."""
        model_config = ModelConfig(name="rerank-model", type="reranker")
        mock_config = MockDataConfig()
        
        generator = RerankerMockGenerator(model_config, mock_config)
        requests = generator.generate(2)
        
        assert len(requests) == 2
        for req in requests:
            assert req.endpoint == "/v1/rerank"
            assert "query" in req.payload
            assert "documents" in req.payload
    
    def test_vision_generator_generate(self):
        """Test vision generator produces valid requests."""
        model_config = ModelConfig(name="vision-model", type="vision")
        mock_config = MockDataConfig()
        
        generator = VisionMockGenerator(model_config, mock_config)
        requests = generator.generate(2)
        
        assert len(requests) == 2
        for req in requests:
            assert req.endpoint == "/v1/chat/completions"
            assert "messages" in req.payload


class TestMetrics:
    """Tests for metrics collection."""
    
    def test_benchmark_metrics_calculate(self):
        """Test metrics calculation."""
        metrics = BenchmarkMetrics(
            model_name="test",
            model_type="chat",
            scenario_name="test_scenario",
            total_requests=5,
            concurrency=2
        )
        
        # Add request metrics
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
        assert metrics.requests_per_sec == 5.0
        assert metrics.avg_latency == 0.2
    
    def test_metrics_collector(self):
        """Test metrics collector."""
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
            success=True
        )
        
        collector.finalize_benchmark(benchmark, duration=0.5)
        
        assert len(collector.benchmarks) == 1
        assert benchmark.successful_requests == 1
    
    def test_metrics_to_dict(self):
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
        
        assert "model_name" in data
        assert "results" in data
        assert "latency" in data


class TestExporters:
    """Tests for result exporters."""
    
    def test_markdown_exporter(self):
        """Test Markdown export."""
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
            
            exporter = MarkdownExporter(tmpdir)
            filepath = exporter.export([metrics])
            
            assert Path(filepath).exists()
            content = Path(filepath).read_text()
            assert "# LLM Benchmark Results" in content
    
    def test_csv_exporter(self):
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
            for path in results.values():
                assert Path(path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
