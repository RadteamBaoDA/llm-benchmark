# Execution Modes

Based on JMeter Thread Group patterns, the tool supports multiple execution modes for comprehensive load testing.

## Basic Modes

| Mode | Description |
|------|-------------|
| `parallel` | All requests launched simultaneously (tests max queue pressure) |
| `controlled` | Uses semaphore to limit concurrent requests |
| `queue_test` | Burst mode to analyze queue behavior (Ollama/vLLM/TGI) |

## JMeter-Style Modes

| Mode | Description | Key Parameters |
|------|-------------|----------------|
| `ramp_up` | Linear ramp-up like JMeter's Thread Group | `ramp_up_time`, `ramp_up_steps`, `hold_time` |
| `stepping` | Stepping Thread Group - add users in discrete steps | `ramp_up_steps`, `hold_time` |
| `spike` | Spike testing - sudden burst then sustain | `spike_multiplier`, `spike_duration` |
| `constant_rate` | Constant Throughput Timer - fixed RPS | `target_rps` |
| `arrivals` | Arrivals Thread Group - control arrival rate | `arrival_rate` |
| `ultimate` | Ultimate Thread Group - complex multi-phase pattern | `stages` |
| `duration` | Run for fixed time, cycling through requests | `duration_seconds` |

## Mode Configuration Examples

### Ramp-Up Mode

Linear increase of concurrent users over time:

```yaml
- name: "ramp_up_test"
  requests: 200
  concurrency: 40
  mode: "ramp_up"
  ramp_up_time: 30.0      # Reach target in 30 seconds
  ramp_up_steps: 10       # 10 steps: +4 users each
  hold_time: 60.0         # Hold at 40 users for 60 seconds
```

### Stepping Thread Group

Add users in discrete steps with hold time:

```yaml
- name: "stepping_load"
  requests: 300
  concurrency: 50
  mode: "stepping"
  ramp_up_steps: 5        # 5 steps: +10 users each
  hold_time: 10.0         # Hold 10 seconds at each level
```

### Spike Testing

Sudden burst of traffic:

```yaml
- name: "spike_test"
  requests: 200
  concurrency: 20          # Base concurrency
  mode: "spike"
  spike_multiplier: 5.0    # 5x spike (20 -> 100 concurrent)
  spike_duration: 10.0     # Spike lasts 10 seconds
```

### Constant Throughput

Fixed requests per second:

```yaml
- name: "constant_throughput"
  requests: 300
  concurrency: 30          # Max concurrent limit
  mode: "constant_rate"
  target_rps: 10.0         # 10 requests per second
```

### Arrivals Thread Group

Control arrival rate (no concurrency limit):

```yaml
- name: "arrivals_test"
  requests: 200
  concurrency: 100         # High limit
  mode: "arrivals"
  arrival_rate: 20.0       # 20 new requests per second
```

### Ultimate Thread Group

Complex multi-phase load pattern:

```yaml
- name: "ultimate_pattern"
  requests: 500
  concurrency: 50
  mode: "ultimate"
  stages:
    - { start: 0, end: 10, duration: 10 }    # Ramp to 10 in 10s
    - { start: 10, end: 30, duration: 15 }   # Ramp to 30 in 15s
    - { start: 30, end: 50, duration: 10 }   # Ramp to 50 in 10s
    - { start: 50, end: 50, duration: 30 }   # Hold at 50 for 30s
    - { start: 50, end: 20, duration: 10 }   # Ramp down to 20
    - { start: 20, end: 0, duration: 10 }    # Ramp down to 0
```

### Duration-Based Testing

Run for fixed time:

```yaml
- name: "duration_test"
  requests: 100            # Requests to cycle through
  concurrency: 15
  mode: "duration"
  duration_seconds: 120.0  # Run for 2 minutes
```
