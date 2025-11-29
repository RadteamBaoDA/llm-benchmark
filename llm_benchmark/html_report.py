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
                <div class="label">Avg Latency</div>
                <div class="value">{avg_latency:.0f}<span class="unit">ms</span></div>
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
            <button class="nav-tab" onclick="showTab('latency')">Latency Analysis</button>
            <button class="nav-tab" onclick="showTab('throughput')">Throughput</button>
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
                <h3>Latency Percentiles</h3>
                <div class="percentile-grid">
                    <div class="percentile-item">
                        <div class="label">Min</div>
                        <div class="value">{min_latency:.0f}ms</div>
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
                    <h3>Response Time Percentiles Over Time</h3>
                    <div class="chart-container">
                        <canvas id="latencyPercentilesChart"></canvas>
                    </div>
                </div>
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
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Total Requests</td>
                            <td>{total_requests}</td>
                        </tr>
                        <tr>
                            <td>Successful Requests</td>
                            <td>{successful_requests}</td>
                        </tr>
                        <tr>
                            <td>Failed Requests</td>
                            <td>{failed_requests}</td>
                        </tr>
                        <tr>
                            <td>Average Requests/Second</td>
                            <td>{requests_per_sec:.2f}</td>
                        </tr>
                        <tr>
                            <td>Total Tokens</td>
                            <td>{total_tokens}</td>
                        </tr>
                        <tr>
                            <td>Average Tokens/Second</td>
                            <td>{tokens_per_sec:.2f}</td>
                        </tr>
                    </tbody>
                </table>
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
        
        // Latency Percentiles Over Time
        new Chart(document.getElementById('latencyPercentilesChart'), {{
            type: 'line',
            data: {{
                labels: latencyOverTimeData.map(d => (d.elapsed_ms / 1000).toFixed(1) + 's'),
                datasets: [{{
                    label: 'Min',
                    data: latencyOverTimeData.map(d => d.min_latency_ms),
                    borderColor: '#16a34a',
                    fill: false,
                    tension: 0.3
                }}, {{
                    label: 'Avg',
                    data: latencyOverTimeData.map(d => d.avg_latency_ms),
                    borderColor: '#2563eb',
                    fill: false,
                    tension: 0.3
                }}, {{
                    label: 'P95',
                    data: latencyOverTimeData.map(d => d.p95_latency_ms),
                    borderColor: '#ea580c',
                    fill: false,
                    tension: 0.3
                }}, {{
                    label: 'Max',
                    data: latencyOverTimeData.map(d => d.max_latency_ms),
                    borderColor: '#dc2626',
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
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to write HTML reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
            "tokens_per_sec": stats.get("tokens_per_sec", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "avg_latency": stats.get("latency_avg_ms", 0),
            "min_latency": stats.get("latency_min_ms", 0),
            "max_latency": stats.get("latency_max_ms", 0),
            "p50_latency": stats.get("latency_p50_ms", 0),
            "p90_latency": stats.get("latency_p90_ms", 0),
            "p95_latency": stats.get("latency_p95_ms", 0),
            "p99_latency": stats.get("latency_p99_ms", 0),
            "latency_over_time_json": json.dumps(latency_over_time),
            "throughput_over_time_json": json.dumps(throughput_over_time),
            "latency_distribution_json": json.dumps(latency_distribution),
            "error_table": error_table,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
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
