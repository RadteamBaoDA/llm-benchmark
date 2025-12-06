# Queue Testing for Inference Servers

The tool is optimized for testing queue behavior of LLM inference servers like Ollama, vLLM, and TGI.

## Queue Metrics

When running benchmarks, the following queue metrics are collected:

| Metric | Description |
|--------|-------------|
| Peak Concurrent | Maximum simultaneous active requests |
| Max Queue Depth | Maximum observed queue depth |
| Avg Queue Depth | Average queue depth during test |
| Avg Wait Time | Average time spent waiting in queue |
| Max Wait Time | Maximum queue wait time |
| Rejections | Requests rejected (HTTP 429/503) |
| Timeouts | Requests that timed out |

## Queue Test Scenario

```yaml
- name: "queue_depth_test"
  description: "Test inference server queue handling"
  requests: 100
  concurrency: 100        # All requests at once
  warmup_requests: 1
  timeout: 180
  mode: "queue_test"
```

## Example Output

```
📋 Queue Metrics:
   Peak Concurrent:           100
   Max Queue Depth:            85
   Avg Queue Depth:         42.30
   Avg Wait Time:        1523.45ms
   Max Wait Time:        8234.12ms
   ⚠️  Rejections (429/503): 5
   ⚠️  Timeouts:             2
```

## Use Cases

1. **Capacity Planning**: Determine maximum concurrent requests your server can handle
2. **Queue Behavior Analysis**: Understand how your inference server queues and processes requests
3. **Timeout Tuning**: Find optimal timeout values for your workload
4. **Bottleneck Identification**: Identify when queue depth becomes a performance issue
