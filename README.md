# LLM Benchmark Tool

A comprehensive benchmarking tool for OpenAI-compatible APIs. This tool supports benchmarking for various model types including chat, embedding, reranker, and vision models.

## Features

- ✅ **Multi-Model Support**: Benchmark chat, embedding, reranker, and vision models
- ✅ **Configurable Load Testing**: Set number of requests and concurrency levels
- ✅ **Batch Scenarios**: Run multiple benchmark scenarios in sequence
- ✅ **Progress Visualization**: Real-time progress bar with tqdm
- ✅ **Response Capture**: Optional capture of full API responses for analysis
- ✅ **Multiple Export Formats**: Export results to Markdown, CSV, and JSON
- ✅ **Comprehensive Metrics**: Measure latency, throughput, and token metrics
- ✅ **YAML Configuration**: All settings configurable via `config.yml`

## Metrics

The benchmark tool measures and reports:

| Metric | Description |
|--------|-------------|
| Successful Requests | Number of requests that completed successfully |
| Failed Requests | Number of requests that failed |
| Duration | Total execution time |
| Requests/sec | Request throughput |
| Tokens/sec | Token generation throughput |
| ns/Token | Nanoseconds per token |
| Average Latency | Mean response time |
| P50 Latency | Median response time |
| P95 Latency | 95th percentile response time |
| P99 Latency | 99th percentile response time |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-bench-tool

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Default Configuration

```bash
python benchmark.py --init
```

This creates a `config.yml` file with default settings.

### 2. Edit Configuration

Edit `config.yml` to set your API endpoint, model, and benchmark parameters.

### 3. Run Benchmark

```bash
# Run with default settings from config
python benchmark.py

# Run all scenarios defined in config
python benchmark.py --scenarios

# Run a specific scenario
python benchmark.py --scenario heavy_load
```

## Configuration

All settings are managed through `config.yml`:

```yaml
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

# Benchmark Scenarios
scenarios:
  - name: "light_load"
    requests: 50
    concurrency: 5
    description: "Light load test"
  
  - name: "medium_load"
    requests: 100
    concurrency: 10
    description: "Medium load test"

# Benchmark Settings
benchmark:
  default_requests: 100
  default_concurrency: 10
  capture_responses: false
  output_dir: "results"
  export_formats:
    - "markdown"
    - "csv"
```

## Command Line Options

```bash
python benchmark.py [OPTIONS]

Options:
  --config, -c PATH       Path to configuration file (default: config.yml)
  --init                  Generate a default config.yml file
  
  # API Settings (override config)
  --base-url URL          API base URL
  --api-key KEY           API key for authentication
  
  # Model Settings (override config)
  --model NAME            Model name to benchmark
  --model-type TYPE       Model type: chat, embed, reranker, vision
  --max-tokens N          Maximum tokens per request
  --temperature FLOAT     Temperature for sampling
  
  # Benchmark Settings
  --requests, -n N        Total number of requests
  --concurrency, -j N     Number of concurrent workers
  
  # Scenario Mode
  --scenarios, -s         Run all scenarios defined in config
  --scenario NAME         Run a specific scenario by name
  
  # Output Settings
  --output-dir, -o DIR    Output directory for results
  --export, -e FORMAT     Export formats: markdown, csv, json
  --capture-responses     Capture full API responses
  
  # Display Settings
  --quiet, -q             Hide progress bar
  --verbose, -v           Verbose output
```

## Usage Examples

### Basic Benchmark

```bash
# Run 100 requests with 10 concurrent workers
python benchmark.py --base-url http://localhost:8000 --model gpt-3.5-turbo --requests 100 --concurrency 10
```

### Benchmark Different Model Types

```bash
# Chat model
python benchmark.py --model-type chat --model gpt-4

# Embedding model
python benchmark.py --model-type embed --model text-embedding-3-small

# Reranker model
python benchmark.py --model-type reranker --model rerank-english-v3.0

# Vision model
python benchmark.py --model-type vision --model gpt-4-vision-preview
```

### Run All Scenarios

```bash
python benchmark.py --scenarios
```

### Export Results

```bash
# Export to multiple formats
python benchmark.py --export markdown csv json

# Specify output directory
python benchmark.py --output-dir ./my-results
```

### Capture Responses

```bash
python benchmark.py --capture-responses
```

## Output Example

```
============================================================
🔥 LLM Benchmark Tool
============================================================

📝 Configuration:
   Base URL:       http://localhost:8000
   Model:          gpt-3.5-turbo
   Model Type:     chat
   Scenarios:      4

🚀 Running scenario: light_load
   Requests: 50, Concurrency: 5

  light_load: 100%|████████████████████| 50/50 [00:05<00:00,  9.12it/s]

============================================================
📊 Results for: light_load
============================================================

✅ 50/50 requests succeeded in 5.48s

📈 Throughput:
   Requests/s:           9.12
   Tokens/s:           285.40
   ns/Token:       3504123.45

⏱️  Latency:
   Average:           0.5234s
   P50 (median):      0.4892s
   P95:               0.8123s
   P99:               0.9567s
   Min:               0.3124s
   Max:               1.0234s

🔤 Tokens:
   Total:              1564
   Prompt:              500
   Completion:         1064
   Avg/Request:        31.28
```

## Export Formats

### Markdown

Creates a detailed report with:
- Summary table of all scenarios
- Detailed results for each scenario
- Configuration details
- Latency and token metrics

### CSV

Creates a tabular format with all metrics:
- One row per scenario
- All configuration and result columns
- Easy to import into spreadsheet applications

### JSON

Creates a structured JSON file with:
- Complete metrics data
- Timestamps and configuration
- Suitable for programmatic analysis

## Project Structure

```
llm-bench-tool/
├── benchmark.py           # Main CLI entry point
├── config.yml             # Configuration file
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── llm_benchmark/         # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration loading
│   ├── engine.py          # Benchmark engine
│   ├── exporters.py       # Export handlers
│   ├── metrics.py         # Metrics collection
│   └── mock_data.py       # Mock data generators
└── results/               # Output directory (created on run)
    ├── benchmark_*.md
    ├── benchmark_*.csv
    └── benchmark_*.json
```

## Supported API Endpoints

The tool uses OpenAI-compatible API endpoints:

| Model Type | Endpoint |
|------------|----------|
| Chat | `/v1/chat/completions` |
| Embed | `/v1/embeddings` |
| Reranker | `/v1/rerank` |
| Vision | `/v1/chat/completions` |

## Mock Data

The tool generates appropriate mock data for each model type:

- **Chat**: Uses configurable prompts
- **Embed**: Uses configurable text samples
- **Reranker**: Uses query + documents for ranking
- **Vision**: Uses image URL/base64 + prompts

## Requirements

- Python 3.10+
- httpx[http2] >= 0.28.0
- PyYAML >= 6.0.2
- numpy >= 2.1.3
- tqdm >= 4.67.1

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.