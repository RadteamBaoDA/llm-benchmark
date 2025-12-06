# Command Line Options

```bash
python benchmark.py [OPTIONS]
```

## Options

### Configuration

| Option | Description |
|--------|-------------|
| `--config, -c PATH` | Path to configuration file (default: config.yml) |
| `--init` | Generate default config.yml and scenario.yml files |

### Report Generation

| Option | Description |
|--------|-------------|
| `--report, -r PATH` | Generate HTML report from timeseries data (file or directory) |
| `--report-output DIR` | Output directory for HTML reports (default: reports) |
| `--no-timeseries` | Disable timeseries data recording during benchmark |

### API Settings (override config)

| Option | Description |
|--------|-------------|
| `--base-url URL` | API base URL |
| `--api-key KEY` | API key for authentication |

### Model Settings (override config)

| Option | Description |
|--------|-------------|
| `--model NAME` | Model name to benchmark |
| `--model-type TYPE` | Model type: chat, embed, reranker, vision |
| `--max-tokens N` | Maximum tokens per request |
| `--temperature FLOAT` | Temperature for sampling |

### Benchmark Settings

| Option | Description |
|--------|-------------|
| `--requests, -n N` | Total number of requests |
| `--concurrency, -j N` | Number of concurrent workers |

### Scenario Mode

| Option | Description |
|--------|-------------|
| `--scenarios, -s` | Run all scenarios defined in config |
| `--scenario NAME` | Run a specific scenario by name |

### Output Settings

| Option | Description |
|--------|-------------|
| `--output-dir, -o DIR` | Output directory for results |
| `--export, -e FORMAT` | Export formats: markdown, csv, json |
| `--capture-responses` | Capture full API responses |

### Display Settings

| Option | Description |
|--------|-------------|
| `--quiet, -q` | Hide progress bar |
| `--verbose, -v` | Verbose output |

### Debug Settings

| Option | Description |
|--------|-------------|
| `--debug, -d` | Enable detailed debug logging (outputs to debug.log) |
| `--debug-console` | Also print debug output to console (requires --debug) |

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
