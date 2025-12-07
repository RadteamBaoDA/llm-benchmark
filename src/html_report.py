"""
HTML Report Generator for LLM Benchmark Results.
Generates JMeter-style interactive HTML reports from timeseries data.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .timeseries import TimeseriesReader, load_all_timeseries


# HTML Template with embedded Chart.js for visualization
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Benchmark Report - {scenario_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #ea580c;
            --danger-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, #1d4ed8 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 8px;
        }}
        
        .header .subtitle {{
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .header .meta {{
            display: flex;
            gap: 24px;
            margin-top: 16px;
            flex-wrap: wrap;
        }}
        
        .header .meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .stat-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
        }}
        
        .stat-card .label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}
        
        .stat-card .value {{
            font-size: 28px;
            font-weight: 600;
            color: var(--text-color);
        }}
        
        .stat-card .unit {{
            font-size: 14px;
            color: var(--text-muted);
            margin-left: 4px;
        }}
        
        .stat-card.success .value {{
            color: var(--success-color);
        }}
        
        .stat-card.warning .value {{
            color: var(--warning-color);
        }}
        
        .stat-card.danger .value {{
            color: var(--danger-color);
        }}
        
        .charts-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 24px;
            margin-bottom: 24px;
        }}
        
        @media (max-width: 768px) {{
            .charts-section {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .chart-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
        }}
        
        .chart-card h3 {{
            font-size: 16px;
            margin-bottom: 16px;
            color: var(--text-color);
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        
        .table-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        
        .table-card h3 {{
            font-size: 16px;
            margin-bottom: 16px;
            color: var(--text-color);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-color);
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{
            background: var(--bg-color);
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .badge-success {{
            background: #dcfce7;
            color: var(--success-color);
        }}
        
        .badge-danger {{
            background: #fee2e2;
            color: var(--danger-color);
        }}
        
        .progress-bar {{
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-bar .fill {{
            height: 100%;
            background: var(--success-color);
            border-radius: 4px;
        }}
        
        .footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 12px;
        }}
        
        .percentile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
        }}
        
        .percentile-item {{
            text-align: center;
            padding: 16px;
            background: var(--bg-color);
            border-radius: 8px;
        }}
        
        .percentile-item .label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        
        .percentile-item .value {{
            font-size: 20px;
            font-weight: 600;
        }}
        
        /* Request/Response styles */
        .request-card {{
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            overflow: hidden;
        }}
        
        .request-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--card-bg);
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
        }}
        
        .request-header:hover {{
            background: #f1f5f9;
        }}
        
        .request-header .request-id {{
            font-weight: 600;
            color: var(--primary-color);
        }}
        
        .request-header .request-meta {{
            display: flex;
            gap: 16px;
            font-size: 13px;
            color: var(--text-muted);
        }}
        
        .request-header .status-badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }}
        
        .request-header .status-badge.success {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .request-header .status-badge.error {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .request-body {{
            display: none;
            padding: 16px;
        }}
        
        .request-body.expanded {{
            display: block;
        }}
        
        .request-section {{
            margin-bottom: 16px;
        }}
        
        .request-section:last-child {{
            margin-bottom: 0;
        }}
        
        .request-section h4 {{
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .request-section pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .no-capture-message {{
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }}
        
        .no-capture-message code {{
            background: var(--bg-color);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13px;
        }}
        
        .nav-tabs {{
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0;
        }}
        
        .nav-tab {{
            padding: 12px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            color: var(--text-muted);
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }}
        
        .nav-tab:hover {{
            color: var(--primary-color);
        }}
        
        .nav-tab.active {{
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
            font-weight: 500;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 LLM Benchmark Report</h1>
            <p class="subtitle">{scenario_name}</p>
            <div class="meta">
                <div class="meta-item">
                    <span>🤖 Model:</span>
                    <strong>{model_name}</strong>
                </div>
                <div class="meta-item">
                    <span>📁 Type:</span>
                    <strong>{model_type}</strong>
                </div>
                <div class="meta-item">
                    <span>🕐 Start:</span>
                    <strong>{start_time}</strong>
                </div>
                <div class="meta-item">
                    <span>⏱️ Duration:</span>
                    <strong>{duration:.2f}s</strong>
                </div>
            </div>
        </div>
        
        <!-- Summary Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Total Requests</div>
                <div class="value">{total_requests}</div>
            </div>
            <div class="stat-card success">
                <div class="label">Success Rate</div>
                <div class="value">{success_rate:.1f}<span class="unit">%</span></div>
            </div>
            <div class="stat-card">
                <div class="label">Requests/Second</div>
                <div class="value">{requests_per_sec:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="label">Tokens/Second</div>
                <div class="value">{tokens_per_sec_display}</div>
            </div>
            <div class="stat-card">
                <div class="label">Avg Latency</div>
                <div class="value">{avg_latency:.0f}<span class="unit">ms</span></div>
            </div>
            <div class="stat-card">
                <div class="label">Latency Std Dev</div>
                <div class="value">{latency_std:.0f}<span class="unit">ms</span></div>
            </div>
            <div class="stat-card">
                <div class="label">P95 Latency</div>
                <div class="value">{p95_latency:.0f}<span class="unit">ms</span></div>
            </div>
            <div class="stat-card">
                <div class="label">P99 Latency</div>
                <div class="value">{p99_latency:.0f}<span class="unit">ms</span></div>
            </div>
        </div>
        
        <!-- Navigation Tabs -->
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('overview')">Overview</button>
            <button class="nav-tab" onclick="showTab('llm-metrics')">LLM Metrics</button>
            <button class="nav-tab" onclick="showTab('latency')">Latency Analysis</button>
            <button class="nav-tab" onclick="showTab('throughput')">Throughput</button>
            <button class="nav-tab" onclick="showTab('requests')">Request/Response</button>
            <button class="nav-tab" onclick="showTab('errors')">Errors</button>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="charts-section">
                <div class="chart-card">
                    <h3>Response Time Over Time</h3>
                    <div class="chart-container">
                        <canvas id="latencyOverTimeChart"></canvas>
                    </div>
                </div>
                <div class="chart-card">
                    <h3>Throughput Over Time</h3>
                    <div class="chart-container">
                        <canvas id="throughputChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="table-card">
                <h3>Latency Statistics (Response Time per Request)</h3>
                <div class="percentile-grid">
                    <div class="percentile-item">
                        <div class="label">Min</div>
                        <div class="value">{min_latency:.0f}ms</div>
                    </div>
                    <div class="percentile-item">
                        <div class="label">Average</div>
                        <div class="value">{avg_latency:.0f}ms</div>
                    </div>
                    <div class="percentile-item">
                        <div class="label">Std Deviation</div>
                        <div class="value">{latency_std:.0f}ms</div>
                    </div>
                    <div class="percentile-item">
                        <div class="label">P50 (Median)</div>
                        <div class="value">{p50_latency:.0f}ms</div>
                    </div>
                    <div class="percentile-item">
                        <div class="label">P90</div>
                        <div class="value">{p90_latency:.0f}ms</div>
                    </div>
                    <div class="percentile-item">
                        <div class="label">P95</div>
                        <div class="value">{p95_latency:.0f}ms</div>
                    </div>
                    <div class="percentile-item">
                        <div class="label">P99</div>
                        <div class="value">{p99_latency:.0f}ms</div>
                    </div>
                    <div class="percentile-item">
                        <div class="label">Max</div>
                        <div class="value">{max_latency:.0f}ms</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- LLM Metrics Tab (BentoML Guide) -->
        <div id="llm-metrics" class="tab-content">
            <div class="table-card" style="margin-bottom: 24px;">
                <h3>📊 Key LLM Inference Metrics (BentoML Guide)</h3>
                <p style="color: var(--text-muted); margin-bottom: 16px;">
                    These metrics follow the <a href="https://bentoml.com/llm/inference-optimization/llm-inference-metrics" target="_blank" style="color: var(--primary-color);">BentoML LLM Inference Metrics Guide</a> for standardized LLM performance measurement.
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Avg (ms)</th>
                            <th>Avg (s)</th>
                            <th>P50 (ms)</th>
                            <th>P95 (ms)</th>
                            <th>P99 (ms)</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>E2EL</strong> (End-to-End Latency)</td>
                            <td>{avg_latency:.0f}</td>
                            <td>{avg_latency_sec:.3f}</td>
                            <td>{p50_latency:.0f}</td>
                            <td>{p95_latency:.0f}</td>
                            <td>{p99_latency:.0f}</td>
                            <td>Total time from request to final token</td>
                        </tr>
                        <tr>
                            <td><strong>TTFT</strong> (Time to First Token)</td>
                            <td>{ttft_avg_display}</td>
                            <td>{ttft_avg_sec_display}</td>
                            <td>{ttft_p50_display}</td>
                            <td>{ttft_p95_display}</td>
                            <td>{ttft_p99_display}</td>
                            <td>Time to generate first token (streaming only)</td>
                        </tr>
                        <tr>
                            <td><strong>TPOT</strong> (Time Per Output Token)</td>
                            <td>{tpot_avg_display}</td>
                            <td>{tpot_avg_sec_display}</td>
                            <td>{tpot_p50_display}</td>
                            <td>{tpot_p95_display}</td>
                            <td>{tpot_p99_display}</td>
                            <td>Average time gap between tokens: (E2EL-TTFT)/(tokens-1)</td>
                        </tr>
                        <tr>
                            <td><strong>ITL</strong> (Inter-Token Latency)</td>
                            <td>{itl_avg_display}</td>
                            <td>{itl_avg_sec_display}</td>
                            <td>-</td>
                            <td>-</td>
                            <td>-</td>
                            <td>Exact pause between consecutive tokens</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="table-card" style="margin-bottom: 24px;">
                <h3>🚀 Throughput Metrics</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>RPS</strong> (Requests Per Second)</td>
                            <td>{requests_per_sec:.2f}</td>
                            <td>How many requests completed per second</td>
                        </tr>
                        <tr>
                            <td><strong>TPS</strong> (Tokens Per Second - Combined)</td>
                            <td>{tokens_per_sec_display}</td>
                            <td>Total tokens (input + output) processed per second</td>
                        </tr>
                        <tr>
                            <td><strong>Input TPS</strong></td>
                            <td>{input_tps_display}</td>
                            <td>Input/prompt tokens processed per second</td>
                        </tr>
                        <tr>
                            <td><strong>Output TPS</strong></td>
                            <td>{output_tps_display}</td>
                            <td>Output/completion tokens generated per second</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="table-card">
                <h3>📈 Token Statistics</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Total Tokens (Combined)</td>
                            <td>{total_tokens_display}</td>
                        </tr>
                        <tr>
                            <td>Total Prompt Tokens (Input)</td>
                            <td>{total_prompt_tokens}</td>
                        </tr>
                        <tr>
                            <td>Total Completion Tokens (Output)</td>
                            <td>{total_completion_tokens}</td>
                        </tr>
                        <tr>
                            <td>Streaming Requests</td>
                            <td>{streaming_requests}</td>
                        </tr>
                        <tr>
                            <td>Non-Streaming Requests</td>
                            <td>{non_streaming_requests}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Latency Analysis Tab -->
        <div id="latency" class="tab-content">
            <div class="charts-section">
                <div class="chart-card">
                    <h3>Response Time Distribution</h3>
                    <div class="chart-container">
                        <canvas id="latencyHistogramChart"></canvas>
                    </div>
                </div>
                <div class="chart-card">
                    <h3>Response Time Percentiles Comparison</h3>
                    <div class="chart-container">
                        <canvas id="latencyPercentilesChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="table-card">
                <h3>Detailed Latency Statistics (Response Time per Request)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Minimum Latency</td>
                            <td>{min_latency:.0f}ms</td>
                            <td>Fastest response time observed</td>
                        </tr>
                        <tr>
                            <td>Average Latency</td>
                            <td>{avg_latency:.0f}ms</td>
                            <td>Mean response time across all requests</td>
                        </tr>
                        <tr>
                            <td>Standard Deviation</td>
                            <td>{latency_std:.0f}ms</td>
                            <td>Variability measure - lower is more consistent</td>
                        </tr>
                        <tr>
                            <td>P50 (Median)</td>
                            <td>{p50_latency:.0f}ms</td>
                            <td>50% of requests completed under this time</td>
                        </tr>
                        <tr>
                            <td>P90</td>
                            <td>{p90_latency:.0f}ms</td>
                            <td>90% of requests completed under this time</td>
                        </tr>
                        <tr>
                            <td>P95</td>
                            <td>{p95_latency:.0f}ms</td>
                            <td>95% of requests completed under this time</td>
                        </tr>
                        <tr>
                            <td>P99</td>
                            <td>{p99_latency:.0f}ms</td>
                            <td>99% of requests completed under this time</td>
                        </tr>
                        <tr>
                            <td>Maximum Latency</td>
                            <td>{max_latency:.0f}ms</td>
                            <td>Slowest response time observed</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Throughput Tab -->
        <div id="throughput" class="tab-content">
            <div class="charts-section">
                <div class="chart-card">
                    <h3>Requests Per Second</h3>
                    <div class="chart-container">
                        <canvas id="rpsChart"></canvas>
                    </div>
                </div>
                <div class="chart-card">
                    <h3>Tokens Per Second</h3>
                    <div class="chart-container">
                        <canvas id="tpsChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="table-card">
                <h3>Throughput Summary</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Total Requests</td>
                            <td>{total_requests}</td>
                            <td>Total number of API requests sent</td>
                        </tr>
                        <tr>
                            <td>Successful Requests</td>
                            <td>{successful_requests}</td>
                            <td>Requests completed with HTTP 2xx status</td>
                        </tr>
                        <tr>
                            <td>Failed Requests</td>
                            <td>{failed_requests}</td>
                            <td>Requests that failed or returned errors</td>
                        </tr>
                        <tr>
                            <td>Test Duration</td>
                            <td>{duration:.2f}s</td>
                            <td>Total time from first to last request</td>
                        </tr>
                        <tr>
                            <td>Requests/Second (RPS)</td>
                            <td>{requests_per_sec:.2f}</td>
                            <td>Average throughput: successful requests per second</td>
                        </tr>
                        <tr>
                            <td>Total Tokens</td>
                            <td>{total_tokens_display}</td>
                            <td>Sum of all tokens processed (prompt + completion)</td>
                        </tr>
                        <tr>
                            <td>Tokens/Second (TPS)</td>
                            <td>{tokens_per_sec_display}</td>
                            <td>Average token throughput per second</td>
                        </tr>
                        <tr>
                            <td>Avg Response Time</td>
                            <td>{avg_latency:.0f}ms</td>
                            <td>Average latency per request (end-to-end)</td>
                        </tr>
                        <tr>
                            <td>Response Time Std Dev</td>
                            <td>{latency_std:.0f}ms</td>
                            <td>Standard deviation - measures consistency</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Request/Response Tab -->
        <div id="requests" class="tab-content">
            <div class="table-card" style="margin-bottom: 24px;">
                <h3>💬 Request/Response Log</h3>
                <p style="color: var(--text-muted); margin-bottom: 12px;">
                    View the prompt sent to the LLM and the response received for each request.
                    {capture_status}
                </p>
                <div class="request-search-bar" style="margin-bottom: 16px;">
                    <input type="text" id="request-search" placeholder="🔍 Search by ID, prompt, or response..." 
                           style="width: 100%; padding: 10px 14px; border: 1px solid var(--border-color); border-radius: 6px; font-size: 14px;" 
                           oninput="filterRequests(this.value)">
                </div>
                <div id="request-list" style="max-height: 50vh; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 6px; padding: 8px;">
                    {request_response_content}
                </div>
                <div id="request-count" style="margin-top: 8px; font-size: 12px; color: var(--text-muted);"></div>
            </div>
        </div>
        
        <!-- Errors Tab -->
        <div id="errors" class="tab-content">
            <div class="chart-card" style="margin-bottom: 24px;">
                <h3>Success vs Failure</h3>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="successFailureChart"></canvas>
                </div>
            </div>
            
            <div class="table-card">
                <h3>Error Details</h3>
                {error_table}
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by LLM Benchmark Tool on {generated_at}</p>
        </div>
    </div>
    
    <script>
        // Chart data
        const latencyOverTimeData = {latency_over_time_json};
        const throughputData = {throughput_over_time_json};
        const latencyDistribution = {latency_distribution_json};
        
        // Tab switching
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.nav-tab').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
        
        // Toggle request/response card
        function toggleRequest(header) {{
            const body = header.nextElementSibling;
            body.classList.toggle('expanded');
        }}
        
        // Filter requests by search query
        function filterRequests(query) {{
            const cards = document.querySelectorAll('.request-card');
            const lowerQuery = query.toLowerCase().trim();
            let visibleCount = 0;
            let totalCount = cards.length;
            
            cards.forEach(card => {{
                if (!lowerQuery) {{
                    card.style.display = 'block';
                    visibleCount++;
                    return;
                }}
                
                const requestId = card.querySelector('.request-id')?.textContent || '';
                const prompt = card.querySelector('.request-section pre')?.textContent || '';
                const response = card.querySelectorAll('.request-section pre')[1]?.textContent || '';
                
                const matches = requestId.toLowerCase().includes(lowerQuery) ||
                               prompt.toLowerCase().includes(lowerQuery) ||
                               response.toLowerCase().includes(lowerQuery);
                
                if (matches) {{
                    card.style.display = 'block';
                    visibleCount++;
                }} else {{
                    card.style.display = 'none';
                }}
            }});
            
            const countEl = document.getElementById('request-count');
            if (countEl) {{
                if (lowerQuery) {{
                    countEl.textContent = `Showing ${{visibleCount}} of ${{totalCount}} requests`;
                }} else {{
                    countEl.textContent = `Total: ${{totalCount}} requests`;
                }}
            }}
        }}
        
        // Initialize request count on page load
        document.addEventListener('DOMContentLoaded', function() {{
            const cards = document.querySelectorAll('.request-card');
            const countEl = document.getElementById('request-count');
            if (countEl && cards.length > 0) {{
                countEl.textContent = `Total: ${{cards.length}} requests`;
            }}
        }});
        
        // Chart.js defaults
        Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
        Chart.defaults.color = '#64748b';
        
        // Response Time Over Time Chart
        new Chart(document.getElementById('latencyOverTimeChart'), {{
            type: 'line',
            data: {{
                labels: latencyOverTimeData.map(d => (d.elapsed_ms / 1000).toFixed(1) + 's'),
                datasets: [{{
                    label: 'Avg Response Time (ms)',
                    data: latencyOverTimeData.map(d => d.avg_latency_ms),
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.3
                }}, {{
                    label: 'P95 Response Time (ms)',
                    data: latencyOverTimeData.map(d => d.p95_latency_ms),
                    borderColor: '#ea580c',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top' }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Response Time (ms)' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Elapsed Time' }}
                    }}
                }}
            }}
        }});
        
        // Throughput Chart
        new Chart(document.getElementById('throughputChart'), {{
            type: 'line',
            data: {{
                labels: throughputData.map(d => (d.elapsed_ms / 1000).toFixed(1) + 's'),
                datasets: [{{
                    label: 'Requests',
                    data: throughputData.map(d => d.requests),
                    borderColor: '#16a34a',
                    backgroundColor: 'rgba(22, 163, 74, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top' }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Requests per Interval' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Elapsed Time' }}
                    }}
                }}
            }}
        }});
        
        // Latency Histogram
        new Chart(document.getElementById('latencyHistogramChart'), {{
            type: 'bar',
            data: {{
                labels: latencyDistribution.bins.map(b => b.toFixed(0) + 'ms'),
                datasets: [{{
                    label: 'Frequency',
                    data: latencyDistribution.counts,
                    backgroundColor: 'rgba(37, 99, 235, 0.7)',
                    borderColor: '#2563eb',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Count' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Response Time (ms)' }}
                    }}
                }}
            }}
        }});
        
        // Latency Percentiles Comparison (Horizontal Bar Chart)
        const percentileData = {{
            min: {min_latency:.2f},
            avg: {avg_latency:.2f},
            p50: {p50_latency:.2f},
            p90: {p90_latency:.2f},
            p95: {p95_latency:.2f},
            p99: {p99_latency:.2f},
            max: {max_latency:.2f}
        }};
        new Chart(document.getElementById('latencyPercentilesChart'), {{
            type: 'bar',
            data: {{
                labels: ['Min', 'Avg', 'P50', 'P90', 'P95', 'P99', 'Max'],
                datasets: [{{
                    label: 'Response Time (ms)',
                    data: [percentileData.min, percentileData.avg, percentileData.p50, percentileData.p90, percentileData.p95, percentileData.p99, percentileData.max],
                    backgroundColor: [
                        'rgba(22, 163, 74, 0.7)',   // Min - green
                        'rgba(37, 99, 235, 0.7)',   // Avg - blue
                        'rgba(59, 130, 246, 0.7)',  // P50 - light blue
                        'rgba(245, 158, 11, 0.7)',  // P90 - amber
                        'rgba(234, 88, 12, 0.7)',   // P95 - orange
                        'rgba(220, 38, 38, 0.7)',   // P99 - red
                        'rgba(127, 29, 29, 0.7)'    // Max - dark red
                    ],
                    borderColor: [
                        '#16a34a', '#2563eb', '#3b82f6', '#f59e0b', '#ea580c', '#dc2626', '#7f1d1d'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.parsed.x.toFixed(2) + ' ms';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Response Time (ms)' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Percentile' }}
                    }}
                }}
            }}
        }});
        
        // RPS Chart
        new Chart(document.getElementById('rpsChart'), {{
            type: 'line',
            data: {{
                labels: throughputData.map(d => (d.elapsed_ms / 1000).toFixed(1) + 's'),
                datasets: [{{
                    label: 'Requests/Second',
                    data: throughputData.map(d => d.requests_per_sec),
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Requests/Second' }}
                    }}
                }}
            }}
        }});
        
        // TPS Chart
        new Chart(document.getElementById('tpsChart'), {{
            type: 'line',
            data: {{
                labels: throughputData.map(d => (d.elapsed_ms / 1000).toFixed(1) + 's'),
                datasets: [{{
                    label: 'Tokens/Second',
                    data: throughputData.map(d => d.tokens_per_sec),
                    borderColor: '#16a34a',
                    backgroundColor: 'rgba(22, 163, 74, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Tokens/Second' }}
                    }}
                }}
            }}
        }});
        
        // Success/Failure Pie Chart
        new Chart(document.getElementById('successFailureChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Successful', 'Failed'],
                datasets: [{{
                    data: [{successful_requests}, {failed_requests}],
                    backgroundColor: ['#16a34a', '#dc2626'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right' }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''

# Index template for multiple scenarios
INDEX_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Benchmark Report - Summary</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #ea580c;
            --danger-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}
        
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, #1d4ed8 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 24px;
            text-align: center;
        }}
        
        .header h1 {{ font-size: 32px; margin-bottom: 8px; }}
        .header .subtitle {{ opacity: 0.9; }}
        
        .scenarios-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        
        .scenario-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .scenario-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        
        .scenario-card h3 {{
            font-size: 18px;
            margin-bottom: 8px;
            color: var(--primary-color);
        }}
        
        .scenario-card .model {{
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 16px;
        }}
        
        .scenario-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}
        
        .stat {{
            text-align: center;
            padding: 12px;
            background: var(--bg-color);
            border-radius: 8px;
        }}
        
        .stat .label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        
        .stat .value {{
            font-size: 18px;
            font-weight: 600;
        }}
        
        .view-btn {{
            display: block;
            width: 100%;
            padding: 12px;
            background: var(--primary-color);
            color: white;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            transition: background 0.2s;
        }}
        
        .view-btn:hover {{
            background: #1d4ed8;
        }}
        
        .footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 LLM Benchmark Report</h1>
            <p class="subtitle">Summary of {scenario_count} Scenarios</p>
        </div>
        
        <div class="scenarios-grid">
            {scenario_cards}
        </div>
        
        <div class="footer">
            <p>Generated by LLM Benchmark Tool on {generated_at}</p>
        </div>
    </div>
</body>
</html>
'''

SCENARIO_CARD_TEMPLATE = '''
<div class="scenario-card">
    <h3>{scenario_name}</h3>
    <p class="model">🤖 {model_name} ({model_type})</p>
    <div class="scenario-stats">
        <div class="stat">
            <div class="label">Requests</div>
            <div class="value">{total_requests}</div>
        </div>
        <div class="stat">
            <div class="label">Success</div>
            <div class="value" style="color: var(--success-color);">{success_rate:.1f}%</div>
        </div>
        <div class="stat">
            <div class="label">Avg Latency</div>
            <div class="value">{avg_latency:.0f}ms</div>
        </div>
    </div>
    <a href="{report_file}" class="view-btn">View Detailed Report →</a>
</div>
'''


class HTMLReportGenerator:
    """Generates JMeter-style HTML reports from timeseries data."""
    
    def __init__(self, output_dir: str = "reports", timeseries_dir: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to write HTML reports
            timeseries_dir: Directory containing timeseries and capture files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeseries_dir = Path(timeseries_dir) if timeseries_dir else None
    
    def generate_scenario_report(self, reader: TimeseriesReader) -> str:
        """
        Generate HTML report for a single scenario.
        
        Args:
            reader: TimeseriesReader with loaded data
            
        Returns:
            Path to generated HTML file
        """
        stats = reader.get_statistics()
        latency_over_time = reader.get_latency_over_time(bucket_size_ms=1000)
        throughput_over_time = reader.get_throughput_over_time(bucket_size_ms=1000)
        latency_distribution = reader.get_latency_distribution(bins=30)
        
        # Generate error table
        error_table = self._generate_error_table(stats.get("errors", {}))
        
        # Handle tokens display - show N/A if not available (e.g., for reranker models)
        total_tokens = stats.get("total_tokens", 0)
        tokens_per_sec = stats.get("tokens_per_sec", 0)
        tokens_per_sec_display = f"{tokens_per_sec:.2f}" if total_tokens > 0 else "N/A"
        total_tokens_display = str(total_tokens) if total_tokens > 0 else "N/A (not reported by API)"
        
        # Input/Output tokens
        input_tps = stats.get("input_tokens_per_sec", 0)
        output_tps = stats.get("output_tokens_per_sec", 0)
        input_tps_display = f"{input_tps:.2f}" if input_tps > 0 else "N/A"
        output_tps_display = f"{output_tps:.2f}" if output_tps > 0 else "N/A"
        
        # TTFT metrics (BentoML guide)
        ttft_avg = stats.get("ttft_avg_ms", 0)
        ttft_p50 = stats.get("ttft_p50_ms", 0)
        ttft_p95 = stats.get("ttft_p95_ms", 0)
        ttft_p99 = stats.get("ttft_p99_ms", 0)
        has_ttft = ttft_avg > 0
        ttft_avg_display = f"{ttft_avg:.0f}" if has_ttft else "N/A"
        ttft_avg_sec_display = f"{ttft_avg / 1000:.3f}" if has_ttft else "N/A"
        ttft_p50_display = f"{ttft_p50:.0f}" if has_ttft else "N/A"
        ttft_p95_display = f"{ttft_p95:.0f}" if has_ttft else "N/A"
        ttft_p99_display = f"{ttft_p99:.0f}" if has_ttft else "N/A"
        
        # TPOT metrics (BentoML guide)
        tpot_avg = stats.get("tpot_avg_ms", 0)
        tpot_p50 = stats.get("tpot_p50_ms", 0)
        tpot_p95 = stats.get("tpot_p95_ms", 0)
        tpot_p99 = stats.get("tpot_p99_ms", 0)
        has_tpot = tpot_avg > 0
        tpot_avg_display = f"{tpot_avg:.2f}" if has_tpot else "N/A"
        tpot_avg_sec_display = f"{tpot_avg / 1000:.4f}" if has_tpot else "N/A"
        tpot_p50_display = f"{tpot_p50:.2f}" if has_tpot else "N/A"
        tpot_p95_display = f"{tpot_p95:.2f}" if has_tpot else "N/A"
        tpot_p99_display = f"{tpot_p99:.2f}" if has_tpot else "N/A"
        
        # ITL metrics (BentoML guide)
        itl_avg = stats.get("itl_avg_ms", 0)
        has_itl = itl_avg > 0
        itl_avg_display = f"{itl_avg:.2f}" if has_itl else "N/A"
        itl_avg_sec_display = f"{itl_avg / 1000:.4f}" if has_itl else "N/A"
        
        # Prompt and completion tokens
        total_prompt_tokens = stats.get("total_prompt_tokens", 0)
        total_completion_tokens = stats.get("total_completion_tokens", 0)
        
        # Streaming stats
        streaming_requests = stats.get("streaming_requests", 0)
        non_streaming_requests = stats.get("non_streaming_requests", 0)
        
        # Prepare template data
        data = {
            "scenario_name": stats.get("scenario_name", "Unknown"),
            "model_name": stats.get("model_name", "Unknown"),
            "model_type": stats.get("model_type", "Unknown"),
            "start_time": stats.get("start_time", ""),
            "duration": stats.get("duration_seconds", 0),
            "total_requests": stats.get("total_requests", 0),
            "successful_requests": stats.get("successful_requests", 0),
            "failed_requests": stats.get("failed_requests", 0),
            "success_rate": stats.get("success_rate", 0),
            "requests_per_sec": stats.get("requests_per_sec", 0),
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_sec_display": tokens_per_sec_display,
            "total_tokens": total_tokens,
            "total_tokens_display": total_tokens_display,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "input_tps_display": input_tps_display,
            "output_tps_display": output_tps_display,
            # E2EL (End-to-End Latency) - same as latency
            "avg_latency": stats.get("latency_avg_ms", 0),
            "latency_std": stats.get("latency_std_ms", 0),
            "min_latency": stats.get("latency_min_ms", 0),
            "max_latency": stats.get("latency_max_ms", 0),
            "p50_latency": stats.get("latency_p50_ms", 0),
            "p90_latency": stats.get("latency_p90_ms", 0),
            "p95_latency": stats.get("latency_p95_ms", 0),
            "p99_latency": stats.get("latency_p99_ms", 0),
            # TTFT (Time to First Token)
            "ttft_avg_display": ttft_avg_display,
            "ttft_avg_sec_display": ttft_avg_sec_display,
            "ttft_p50_display": ttft_p50_display,
            "ttft_p95_display": ttft_p95_display,
            "ttft_p99_display": ttft_p99_display,
            "has_ttft": has_ttft,
            # TPOT (Time Per Output Token)
            "tpot_avg_display": tpot_avg_display,
            "tpot_avg_sec_display": tpot_avg_sec_display,
            "tpot_p50_display": tpot_p50_display,
            "tpot_p95_display": tpot_p95_display,
            "tpot_p99_display": tpot_p99_display,
            "has_tpot": has_tpot,
            # ITL (Inter-Token Latency)
            "itl_avg_display": itl_avg_display,
            "itl_avg_sec_display": itl_avg_sec_display,
            "has_itl": has_itl,
            # E2EL in seconds
            "avg_latency_sec": stats.get("latency_avg_ms", 0) / 1000,
            # Streaming
            "streaming_requests": streaming_requests,
            "non_streaming_requests": non_streaming_requests,
            # Charts data
            "latency_over_time_json": json.dumps(latency_over_time),
            "throughput_over_time_json": json.dumps(throughput_over_time),
            "latency_distribution_json": json.dumps(latency_distribution),
            "error_table": error_table,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Load captured request/response data
        # First try timeseries_dir (where capture files are saved)
        capture_file = None
        if self.timeseries_dir:
            capture_file = self._find_capture_file(
                stats.get("scenario_name", ""),
                stats.get("model_name", ""),
                self.timeseries_dir
            )
        # Also try the output directory parent (for legacy compatibility)
        if not capture_file:
            capture_file = self._find_capture_file(
                stats.get("scenario_name", ""),
                stats.get("model_name", ""),
                self.output_dir.parent
            )
        # Also try the output directory itself
        if not capture_file:
            capture_file = self._find_capture_file(
                stats.get("scenario_name", ""),
                stats.get("model_name", ""),
                self.output_dir
            )
        capture_status, request_response_content = self._generate_request_response_content(capture_file)
        data["capture_status"] = capture_status
        data["request_response_content"] = request_response_content
        
        # Render template
        html = HTML_TEMPLATE.format(**data)
        
        # Write file
        safe_name = stats.get("scenario_name", "unknown").replace(" ", "_").replace("/", "-")
        filename = f"report_{safe_name}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)
    
    def _generate_error_table(self, errors: Dict[str, int]) -> str:
        """Generate HTML table for errors."""
        if not errors:
            return "<p style='color: var(--success-color);'>✅ No errors recorded</p>"
        
        rows = ""
        for error, count in sorted(errors.items(), key=lambda x: -x[1]):
            rows += f"<tr><td>{error}</td><td>{count}</td></tr>"
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Error Message</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """
    
    def _find_capture_file(self, scenario_name: str, model_name: str, timeseries_dir: Path) -> Optional[str]:
        """Find the capture file matching the scenario and model."""
        import glob
        
        # Clean names for file matching
        safe_scenario = scenario_name.replace(" ", "_").replace("/", "-").replace(":", "-")
        safe_model = model_name.replace("/", "-").replace(":", "-")
        
        # Search for capture files
        pattern = str(timeseries_dir / f"capture_{safe_scenario}_{safe_model}_*.json")
        files = sorted(glob.glob(pattern), reverse=True)  # Most recent first
        
        if files:
            return files[0]
        return None
    
    def _generate_request_response_content(self, capture_file: Optional[str]) -> tuple:
        """Generate HTML content for request/response viewer."""
        if not capture_file or not Path(capture_file).exists():
            status = "Enable <code>capture_request_response: true</code> in config.yml to capture request/response data."
            content = '<div class="no-capture-message"><p>No captured data available.</p></div>'
            return status, content
        
        try:
            with open(capture_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            status = f"Error loading capture file: {e}"
            content = '<div class="no-capture-message"><p>Failed to load captured data.</p></div>'
            return status, content
        
        if not data:
            status = "Capture file is empty."
            content = '<div class="no-capture-message"><p>No requests were captured.</p></div>'
            return status, content
        
        status = f"Showing {len(data)} captured request(s). Click on a request to expand."
        
        cards = []
        for i, item in enumerate(data):
            request_id = item.get("request_id", i + 1)
            latency_ms = item.get("latency_ms", 0)
            success = item.get("success", False)
            tokens = item.get("tokens", 0)
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            error = item.get("error", "")
            
            status_class = "success" if success else "error"
            status_text = "Success" if success else "Failed"
            
            # Escape HTML in prompt and response
            prompt_escaped = self._escape_html(prompt) or "(Empty prompt)"
            if success:
                response_escaped = self._escape_html(response) or "(Empty response)"
            else:
                response_escaped = self._escape_html(error or "No response")
            
            # Truncate prompt preview for header
            prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            prompt_preview_escaped = self._escape_html(prompt_preview)
            
            card = f'''
            <div class="request-card" data-request-id="{request_id}">
                <div class="request-header" onclick="toggleRequest(this)">
                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        <span class="request-id">Request #{request_id}</span>
                        <span style="font-size: 12px; color: var(--text-muted); max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{prompt_preview_escaped}</span>
                    </div>
                    <div class="request-meta">
                        <span>⏱️ {latency_ms:.0f}ms</span>
                        <span>📝 {tokens} tokens</span>
                        <span class="status-badge {status_class}">{status_text}</span>
                    </div>
                </div>
                <div class="request-body">
                    <div class="request-section">
                        <h4>📤 Prompt</h4>
                        <pre>{prompt_escaped}</pre>
                    </div>
                    <div class="request-section">
                        <h4>📥 Response</h4>
                        <pre>{response_escaped}</pre>
                    </div>
                </div>
            </div>
            '''
            cards.append(card)
        
        content = "\n".join(cards)
        return status, content
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))
    
    def generate_index(self, readers: List[TimeseriesReader]) -> str:
        """
        Generate index page for multiple scenario reports.
        
        Args:
            readers: List of TimeseriesReader instances
            
        Returns:
            Path to generated index HTML file
        """
        scenario_cards = ""
        
        for reader in readers:
            stats = reader.get_statistics()
            safe_name = stats.get("scenario_name", "unknown").replace(" ", "_").replace("/", "-")
            report_file = f"report_{safe_name}.html"
            
            card = SCENARIO_CARD_TEMPLATE.format(
                scenario_name=stats.get("scenario_name", "Unknown"),
                model_name=stats.get("model_name", "Unknown"),
                model_type=stats.get("model_type", "Unknown"),
                total_requests=stats.get("total_requests", 0),
                success_rate=stats.get("success_rate", 0),
                avg_latency=stats.get("latency_avg_ms", 0),
                report_file=report_file
            )
            scenario_cards += card
        
        html = INDEX_TEMPLATE.format(
            scenario_count=len(readers),
            scenario_cards=scenario_cards,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        filepath = self.output_dir / "index.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)
    
    def generate_from_directory(self, timeseries_dir: str) -> str:
        """
        Generate reports for all timeseries files in a directory.
        
        Args:
            timeseries_dir: Directory containing timeseries CSV/JSONL files
            
        Returns:
            Path to generated index HTML file
        """
        # Store timeseries_dir for capture file lookup
        self.timeseries_dir = Path(timeseries_dir)
        
        readers = load_all_timeseries(timeseries_dir)
        
        if not readers:
            raise ValueError(f"No timeseries files found in {timeseries_dir}")
        
        print(f"📊 Found {len(readers)} timeseries file(s)")
        
        # Generate individual reports
        for reader in readers:
            report_path = self.generate_scenario_report(reader)
            print(f"   ✅ Generated: {report_path}")
        
        # Generate index
        index_path = self.generate_index(readers)
        print(f"   📋 Index: {index_path}")
        
        return index_path
    
    def generate_from_files(self, files: List[str]) -> str:
        """
        Generate reports from specific timeseries files.
        
        Args:
            files: List of paths to timeseries files
            
        Returns:
            Path to generated index HTML file
        """
        readers = []
        for file_path in files:
            try:
                readers.append(TimeseriesReader(file_path))
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        if not readers:
            raise ValueError("No valid timeseries files provided")
        
        # Generate individual reports
        for reader in readers:
            report_path = self.generate_scenario_report(reader)
            print(f"   ✅ Generated: {report_path}")
        
        # Generate index
        index_path = self.generate_index(readers)
        print(f"   📋 Index: {index_path}")
        
        return index_path


def generate_html_report(
    input_path: str,
    output_dir: str = "reports"
) -> str:
    """
    Generate HTML reports from timeseries data.
    
    Args:
        input_path: Path to timeseries file or directory
        output_dir: Directory to write HTML reports
        
    Returns:
        Path to generated index HTML file
    """
    generator = HTMLReportGenerator(output_dir)
    
    path = Path(input_path)
    
    if path.is_dir():
        return generator.generate_from_directory(str(path))
    elif path.is_file():
        return generator.generate_from_files([str(path)])
    else:
        raise ValueError(f"Invalid input path: {input_path}")
