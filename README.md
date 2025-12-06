# LLM Benchmark Tool

A comprehensive benchmarking tool for OpenAI-compatible APIs. This tool supports benchmarking for various model types including chat, embedding, reranker, and vision models.

## Features

- ✅ **Multi-Model Support**: Benchmark chat, embedding, reranker, and vision models
- ✅ **Configurable Load Testing**: Set number of requests and concurrency levels
- ✅ **Batch Scenarios**: Run multiple benchmark scenarios in sequence
- ✅ **JMeter-Style Execution Modes**: Parallel, ramp-up, stepping, spike, constant rate, arrivals, ultimate thread group
- ✅ **Queue Testing**: Analyze inference server queue behavior (Ollama, vLLM, TGI)
- ✅ **Progress Visualization**: Real-time progress bar with tqdm
- ✅ **Multiple Export Formats**: Export results to Markdown, CSV, and JSON
- ✅ **Comprehensive Metrics**: Measure latency, throughput, and token metrics
- ✅ **HTML Reports**: Generate JMeter-style interactive HTML reports with charts
- ✅ **YAML Configuration**: All settings configurable via `config.yml` and `scenario.yml`

## Metrics

| Metric | Description |
|--------|-------------|
| Successful/Failed Requests | Request completion status |
| Requests/sec | Request throughput |
| Tokens/sec | Token generation throughput |
| Average Latency | Mean response time |
| P50/P95/P99 Latency | Percentile response times |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-bench-tool

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Default Configuration

```bash
python benchmark.py --init
```

### 2. Edit Configuration

Edit `config.yml` for API endpoint and model settings.
Edit `scenario.yml` for benchmark scenarios.

### 3. Run Benchmark

```bash
# Run with default settings
python benchmark.py

# Run all scenarios
python benchmark.py --scenarios

# Run a specific scenario
python benchmark.py --scenario heavy_load
```

### 4. Generate HTML Report

```bash
python benchmark.py --report results/
```

## Project Structure

```
llm-bench-tool/
├── benchmark.py           # Main CLI entry point
├── config.yml             # Main configuration file
├── scenario.yml           # Benchmark scenarios
├── llm_benchmark/         # Main package
├── tests/                 # Unit tests
├── results/               # Output directory
└── reports/               # HTML reports directory
```

## Requirements

- Python 3.10+
- httpx[http2] >= 0.28.0
- PyYAML >= 6.0.2
- numpy >= 2.1.3
- tqdm >= 4.67.1

## Documentation

For detailed documentation, see the [docs](./docs/) folder:

| Document | Description |
|----------|-------------|
| [Configuration](./docs/configuration.md) | YAML configuration files and API endpoints |
| [Execution Modes](./docs/execution-modes.md) | JMeter-style load testing modes |
| [CLI Options](./docs/cli-options.md) | Command line options and usage examples |
| [Queue Testing](./docs/queue-testing.md) | Inference server queue analysis |
| [HTML Reports](./docs/html-reports.md) | Interactive report generation |
| [Export Formats](./docs/export-formats.md) | Markdown, CSV, JSON exports |
| [Debug Logging](./docs/debug-logging.md) | Debug mode and troubleshooting |

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.