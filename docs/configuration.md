# Configuration

Settings are split into two files for better organization and flexibility.

## config.yml - Main Configuration

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

# Logging Configuration - Enable debug logging to diagnose request failures
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_requests: false  # Log HTTP request details (URL, headers, payload)
  log_responses: false  # Log HTTP response details (status, body)
  log_file: null  # Log file path (null = console only)
  max_payload_length: 500  # Truncate payload in logs
  max_response_length: 500  # Truncate response body in logs

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

## Logging Configuration

The logging section allows you to enable detailed HTTP request/response logging to diagnose issues:

| Setting | Description | Default |
|---------|-------------|---------|
| `level` | Log level: DEBUG, INFO, WARNING, ERROR | `INFO` |
| `log_requests` | Log HTTP request details (URL, headers, payload) | `false` |
| `log_responses` | Log HTTP response details (status code, body) | `false` |
| `log_file` | Path to log file (null = console only) | `null` |
| `max_payload_length` | Maximum characters for payload in logs | `500` |
| `max_response_length` | Maximum characters for response body in logs | `500` |

### Debugging Failed Requests

To diagnose why requests are failing, enable debug logging:

```yaml
logging:
  level: "DEBUG"
  log_requests: true
  log_responses: true
  log_file: "benchmark.log"  # Optional: save to file
```

This will show:
- Full request URL and endpoint
- Request headers (API key masked)
- Request payload
- Response status code
- Response body (with error details)
- Error messages and exception details

## scenario.yml - Benchmark Scenarios

```yaml
# Default settings for all scenarios
defaults:
  warmup_requests: 1
  timeout: 60
  enabled: true
  mode: "parallel"

# Benchmark Scenarios
scenarios:
  - name: "light_load"
    requests: 50
    concurrency: 5
    description: "Light load test"
    mode: "controlled"  # Semaphore-limited concurrency
  
  - name: "medium_load"
    requests: 100
    concurrency: 10
    description: "Medium load test"
    mode: "parallel"  # All requests at once
  
  - name: "ramp_up_test"
    requests: 200
    concurrency: 40
    description: "JMeter-style ramp-up"
    mode: "ramp_up"
    ramp_up_time: 30.0    # 30s to reach full concurrency
    ramp_up_steps: 10     # 10 steps during ramp
    hold_time: 60.0       # Hold at target for 60s
  
  - name: "spike_test"
    requests: 200
    concurrency: 20
    description: "Spike testing"
    mode: "spike"
    spike_multiplier: 5.0  # 5x spike (20 -> 100)
    spike_duration: 10.0   # Spike lasts 10 seconds
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
