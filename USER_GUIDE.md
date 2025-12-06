# LLM Benchmark Tool User Guide

Welcome to the LLM Benchmark Tool! This guide is designed to help you understand how to use this tool to test the performance of Large Language Models (LLMs) and API servers.

Whether you are a beginner or an experienced engineer, this guide will walk you through the concepts, configuration, and reporting features step-by-step.

---

## 1. Quick Start for Beginners

If you just want to run a test and see what happens, follow these steps:

1.  **Initialize the tool**: This creates the necessary configuration files.
    ```bash
    python benchmark.py --init
    ```

2.  **Run a simple test**: This runs the default scenarios defined in `scenario.yml`.
    ```bash
    python benchmark.py --scenarios
    ```

3.  **View the Report**: After the test finishes, generate a visual HTML report.
    ```bash
    python benchmark.py --report results/
    ```
    Open the generated `.html` file in your browser to see charts and graphs.

---

## 2. Understanding Execution Modes

The tool uses different "modes" to simulate how users interact with your system. Think of these as different "traffic patterns."

### Basic Modes (Simple & Easy)

| Mode | What it does | Real-world Analogy |
|------|--------------|-------------------|
| **`parallel`** | Sends all requests at the exact same time. | Everyone trying to buy concert tickets the second they go on sale. |
| **`controlled`** | Limits how many requests happen at once. If you set concurrency to 5, only 5 requests run at a time. | A store that only lets 5 people in at a time. As one leaves, another enters. |
| **`queue_test`** | Sends a burst of requests to see how the server queues them up. | A sudden rush of people joining a waiting line. |

### Advanced Modes (JMeter-Style)

These modes are for more complex testing, similar to tools like JMeter or K6.

| Mode | What it does | Key Settings | Example Scenario |
|------|--------------|--------------|------------------|
| **`ramp_up`** | Starts with 0 users and gradually adds more until it reaches the target. | **Ramp Up Time**: How long to take to reach full load.<br>**Hold Time**: How long to stay at full load. | **Morning Login**: Employees starting work between 8:50 and 9:10 AM. |
| **`stepping`** | Adds users in "steps" (e.g., 10 users, wait 1 min, then 20 users...). | **Steps**: How many jumps to make.<br>**Hold Time**: How long to wait at each step. | **Server Warmup**: Safely warming up a cold cache or testing auto-scaling triggers. |
| **`spike`** | Runs normal traffic, then suddenly hits the server with a huge load, then goes back to normal. | **Spike Multiplier**: How much bigger the spike is (e.g., 5x normal).<br>**Spike Duration**: How long the spike lasts. | **Super Bowl Ad**: Millions of users visiting immediately after a TV spot. |
| **`constant_rate`** | Forces a specific number of requests per second (RPS), no matter how slow the server is. | **Target RPS**: Requests Per Second to send. | **API Contract**: Verifying you can handle the guaranteed 100 RPS for a client. |
| **`ultimate`** | Fully customizable. You define exactly when users start and stop. | **Stages**: A list of steps defining start count, end count, and duration. | **Full Day Simulation**: Morning rush, lunch dip, afternoon peak, evening quiet. |

### 🎯 How to Choose the Right Mode for Your Business Goal

Not sure which mode to pick? Match your business goal to the recommended mode:

| Business Goal | Recommended Mode | Why? |
|---------------|------------------|------|
| **"I want to know the absolute maximum limits of my server."** | `parallel` or `ramp_up` | Pushes the system to failure to find the breaking point. |
| **"I want to simulate normal day-to-day usage."** | `controlled` | Mimics a steady stream of users where new ones arrive as others leave. |
| **"We are launching a marketing campaign/TV ad."** | `spike` | Simulates a sudden flood of users hitting the site at once. |
| **"I need to test if my auto-scaling works correctly."** | `stepping` | Increases load in steps (e.g., +10 users every minute) to give auto-scalers time to react. |
| **"I have a strict SLA (e.g., must handle 50 req/sec)."** | `constant_rate` | Forces a specific throughput to verify if the system maintains performance under contract load. |

---

## 3. Creating a Scenario

A "Scenario" is a test plan. You define these in the `scenario.yml` file.

### Basic Structure
Here is a simple example with explanations:

```yaml
scenarios:
  - name: "daily_usage_test"      # A unique name for this test
    description: "Simulate normal daily traffic"
    requests: 100                 # Total number of questions to ask the LLM
    concurrency: 5                # Max number of users at the same time
    mode: "controlled"            # The mode to use (see section 2)
    timeout: 60                   # Stop waiting if no answer after 60 seconds
    enabled: true                 # Set to 'false' to skip this test
```

### Advanced Examples

**Scenario 1: The "Morning Rush" (Ramp Up)**
Simulate users logging in gradually over 30 seconds.
```yaml
  - name: "morning_rush"
    mode: "ramp_up"
    requests: 200
    concurrency: 20         # Target: 20 active users
    ramp_up_time: 30.0      # Take 30s to get to 20 users
    hold_time: 60.0         # Stay at 20 users for 1 minute
```

**Scenario 2: The "Viral Post" (Spike)**
Simulate a sudden viral event where traffic jumps 5x.
```yaml
  - name: "viral_event"
    mode: "spike"
    requests: 500
    concurrency: 10         # Normal load: 10 users
    spike_multiplier: 5.0   # Spike load: 50 users (10 * 5)
    spike_duration: 15.0    # The spike lasts 15 seconds
```

---

## 4. Reporting & Analysis

After running your tests, you need to understand the results. The tool provides three ways to view data.

### 1. HTML Report (Best for Visuals)
This generates an interactive dashboard with charts.
- **Command**: `python benchmark.py --report results/`
- **Features**:
    - **Latency Charts**: See how response time changed over the test.
    - **Throughput**: See Requests Per Second (RPS) and Tokens Per Second (TPS).
    - **Error Analysis**: See exactly what errors occurred and how often.
    - **Percentiles**: Check P95 and P99 latency (e.g., "99% of requests were faster than X").

### 2. CSV & JSON Data (Best for Data Analysis)
Raw data is saved in the `results/` folder.
- **CSV (`.csv`)**: Open in Excel or Google Sheets. Contains one row per test run with summary stats.
- **JSON (`.json`)**: Good for programmatic analysis or sending to other tools.
- **Timeseries Data**: Inside `results/timeseries/`, you will find detailed logs of *every single request*. You can load this into Python (Pandas) or Tableau for deep analysis.

### 3. Markdown Summary (Best for Sharing)
A simple text summary (`.md`) is generated in `results/`. You can copy-paste this directly into GitHub issues, Slack, or emails to share findings with your team.

---

## 5. Configuration (`config.yml`)

The `config.yml` file is where you set up your connection to the LLM.

```yaml
api:
  base_url: "http://localhost:8000"  # The address of your LLM server
  api_key: "your-key-here"           # If your server needs a password/key
  timeout: 60                        # Max time to wait for a response
  streaming: true                    # Enable streaming for TTFT/TPOT metrics
  endpoint_prefix: "/v1"             # API path prefix (see below)

model:
  name: "gpt-3.5-turbo"              # The model name to send in requests
  type: "chat"                       # Usually "chat" for modern LLMs
  max_tokens: 100                    # Limit response length (saves time/money)

benchmark:
  output_dir: "results"              # Where to save reports
  export_formats: ["markdown", "csv", "json"] # Which report formats to create
```

### Endpoint Prefix for Different Providers

The `endpoint_prefix` setting controls the API path prefix. Different LLM providers use different conventions:

| Provider | `endpoint_prefix` | Full URL Example |
|----------|-------------------|------------------|
| **OpenAI** | `/v1` (default) | `https://api.openai.com/v1/chat/completions` |
| **Ollama** | `/v1` | `http://localhost:11434/v1/chat/completions` |
| **vLLM** | `/v1` | `http://localhost:8000/v1/chat/completions` |
| **LiteLLM Proxy** | `""` (empty) | `http://localhost:4000/chat/completions` |
| **Azure OpenAI** | `/openai/deployments/YOUR_DEPLOYMENT` | Custom path |

**Example for LiteLLM Proxy:**
```yaml
api:
  base_url: "http://localhost:4000"
  endpoint_prefix: ""    # No prefix for LiteLLM proxy
```

---

## 6. Troubleshooting & Debugging

If things aren't working as expected:

**1. Enable Debug Logging**
See exactly what the tool is doing by turning on debug mode.
```bash
python benchmark.py --scenarios --debug --debug-console
```
This will print detailed logs to your screen, showing every request sent and response received.

**2. Check `debug.log`**
A file named `debug.log` is created in your folder. It contains even more technical details than the console output.

**3. Common Issues**
- **Connection Refused**: Check if your `base_url` is correct and your server is running.
- **401 Unauthorized**: Check your `api_key` in `config.yml`.
- **Timeout**: Increase the `timeout` value in `config.yml` if your model is slow.
