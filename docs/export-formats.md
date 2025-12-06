# Export Formats

The benchmark tool supports multiple export formats for results analysis.

## Usage

```bash
# Export to multiple formats
python benchmark.py --export markdown csv json

# Specify output directory
python benchmark.py --output-dir ./my-results
```

## Markdown

Creates a detailed report with:
- Summary table of all scenarios
- Detailed results for each scenario
- Configuration details
- Latency and token metrics

## CSV

Creates a tabular format with all metrics:
- One row per scenario
- All configuration and result columns
- Easy to import into spreadsheet applications

## JSON

Creates a structured JSON file with:
- Complete metrics data
- Timestamps and configuration
- Suitable for programmatic analysis

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
