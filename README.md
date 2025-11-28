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
- ✅ **YAML Configuration**: All settings configurable via `config.yml` and `scenario.yml`
- ✅ **Timeseries Recording**: Record per-request metrics with timestamps
- ✅ **HTML Reports**: Generate JMeter-style interactive HTML reports with charts
- ✅ **Local Image Support**: Use local image files for vision model testing

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

This creates a `config.yml` and `scenario.yml` file with default settings.

### 2. Edit Configuration

Edit `config.yml` to set your API endpoint and model settings.
Edit `scenario.yml` to define your benchmark scenarios.

### 3. Run Benchmark

```bash
# Run with default settings from config
python benchmark.py

# Run all scenarios defined in scenario.yml
python benchmark.py --scenarios

# Run a specific scenario
python benchmark.py --scenario heavy_load
```

### 4. Generate HTML Report

```bash
# Generate interactive HTML report from timeseries data
python benchmark.py --report results/
```

## Configuration

Settings are split into two files:

### config.yml - Main Configuration

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

# Mock Data Configuration
mock_data:
  prompts:
    - "Explain quantum computing in simple terms."
    - "Write a short story about a robot."
  vision_image_path: "./images/test.jpg"  # Single local image
  vision_image_paths:  # Or multiple images (cycles through)
    - "./images/image1.jpg"
    - "./images/image2.png"

# Reference to scenarios file
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
```

### scenario.yml - Benchmark Scenarios

```yaml
# Default settings for all scenarios
defaults:
  warmup_requests: 1
  timeout: 60
  enabled: true

# Benchmark Scenarios
scenarios:
  - name: "light_load"
    requests: 50
    concurrency: 5
    description: "Light load test"
    warmup_requests: 2
  
  - name: "medium_load"
    requests: 100
    concurrency: 10
    description: "Medium load test"
    timeout: 120
  
  - name: "heavy_load"
    requests: 500
    concurrency: 50
    description: "Heavy load stress test"
    enabled: true  # Set to false to skip
```

## Command Line Options

```bash
python benchmark.py [OPTIONS]

Options:
  --config, -c PATH       Path to configuration file (default: config.yml)
  --init                  Generate default config.yml and scenario.yml files
  
  # Report Generation
  --report, -r PATH       Generate HTML report from timeseries data (file or directory)
  --report-output DIR     Output directory for HTML reports (default: reports)
  --no-timeseries         Disable timeseries data recording during benchmark
  
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

### Generate HTML Reports

```bash
# Generate report from all timeseries files in results/
python benchmark.py --report results/

# Generate report from specific timeseries file
python benchmark.py --report results/timeseries_scenario1_gpt-4_20231115.csv

# Specify custom output directory for reports
python benchmark.py --report results/ --report-output my_reports/

# Run benchmark without timeseries recording
python benchmark.py --scenarios --no-timeseries
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
   Warmup: 1, Timeout: 60s
   📊 Timeseries: results/timeseries_light_load_gpt-3.5-turbo_20231115_120000.csv

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

## HTML Reports

The tool generates interactive JMeter-style HTML reports with:

### Dashboard Features
- **Summary Statistics**: Total requests, success rate, throughput, latency metrics
- **Tabbed Interface**: Overview, Latency Analysis, Throughput, Errors
- **Interactive Charts**: Powered by Chart.js

### Charts Included
| Chart | Description |
|-------|-------------|
| Response Time Over Time | Average and P95 latency trends |
| Throughput Over Time | Requests completed per time bucket |
| Latency Distribution | Histogram of response times |
| Latency Percentiles | Min/Avg/P95/Max over time |
| Requests/Second | RPS trend during benchmark |
| Tokens/Second | Token throughput trend |
| Success/Failure | Pie chart of request outcomes |

### Percentile Summary
- Min, P50 (Median), P90, P95, P99, Max latencies

### Error Analysis
- Detailed error table with counts
- Breakdown by error type

### Multi-Scenario Index
When multiple scenarios are benchmarked, an `index.html` is generated with:
- Summary cards for each scenario
- Quick stats (requests, success rate, avg latency)
- Links to detailed individual reports

## Timeseries Data

During benchmark execution, per-request metrics are recorded to CSV files:

```csv
timestamp,elapsed_ms,scenario_name,model_name,model_type,request_id,latency_ms,success,tokens,...
1700000000.123,0,light_load,gpt-4,chat,1,152.3,True,45,...
1700000000.275,152,light_load,gpt-4,chat,2,148.7,True,52,...
```

### Recorded Fields
| Field | Description |
|-------|-------------|
| timestamp | Unix timestamp of request completion |
| elapsed_ms | Time since benchmark start |
| scenario_name | Name of the scenario |
| model_name | Model being tested |
| request_id | Sequential request number |
| latency_ms | Request latency in milliseconds |
| success | Whether request succeeded |
| tokens | Total tokens in response |
| concurrent_requests | Active concurrent requests |

## Project Structure

```
llm-bench-tool/
├── benchmark.py           # Main CLI entry point
├── config.yml             # Main configuration file
├── scenario.yml           # Benchmark scenarios
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
├── README.md              # This file
├── llm_benchmark/         # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration loading
│   ├── engine.py          # Benchmark engine
│   ├── exporters.py       # Export handlers (MD, CSV, JSON)
│   ├── metrics.py         # Metrics collection
│   ├── mock_data.py       # Mock data generators
│   ├── timeseries.py      # Timeseries recording/reading
│   └── html_report.py     # HTML report generator
├── tests/                 # Unit tests
│   └── test_benchmark.py
├── results/               # Output directory (created on run)
│   ├── benchmark_*.md
│   ├── benchmark_*.csv
│   ├── benchmark_*.json
│   └── timeseries_*.csv   # Per-request metrics
└── reports/               # HTML reports directory
    ├── index.html         # Multi-scenario index
    └── report_*.html      # Individual scenario reports
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

- **Chat**: Uses configurable prompts from `mock_data.prompts`
- **Embed**: Uses configurable text samples from `mock_data.texts`
- **Reranker**: Uses query + documents for ranking
- **Vision**: Supports multiple image sources:
  - URL: `mock_data.vision_image_url`
  - Base64: `mock_data.vision_image_base64`
  - Local file: `mock_data.vision_image_path`
  - Multiple local files: `mock_data.vision_image_paths` (cycles through)

## Requirements

- Python 3.10+
- httpx[http2] >= 0.28.0
- PyYAML >= 6.0.2
- numpy >= 2.1.3
- tqdm >= 4.67.1

### Development Requirements

- pytest >= 8.0.0
- pytest-asyncio >= 0.24.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.