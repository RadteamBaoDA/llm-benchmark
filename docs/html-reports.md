# HTML Reports

The tool generates interactive JMeter-style HTML reports with comprehensive visualization.

## Generating Reports

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

## Dashboard Features

- **Summary Statistics**: Total requests, success rate, throughput, latency metrics
- **Tabbed Interface**: Overview, Latency Analysis, Throughput, Errors
- **Interactive Charts**: Powered by Chart.js

## Charts Included

| Chart | Description |
|-------|-------------|
| Response Time Over Time | Average and P95 latency trends |
| Throughput Over Time | Requests completed per time bucket |
| Latency Distribution | Histogram of response times |
| Latency Percentiles | Min/Avg/P95/Max over time |
| Requests/Second | RPS trend during benchmark |
| Tokens/Second | Token throughput trend |
| Success/Failure | Pie chart of request outcomes |

## Percentile Summary

The reports include detailed percentile analysis:
- Min, P50 (Median), P90, P95, P99, Max latencies

## Error Analysis

- Detailed error table with counts
- Breakdown by error type

## Multi-Scenario Index

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
