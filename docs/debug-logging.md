# Debug Logging

The tool includes comprehensive debug logging to help verify execution modes work correctly and trace issues during testing.

## Enable Debug Mode

```bash
# Basic debug logging to file
python benchmark.py --scenarios --debug

# Debug logging to both file and console
python benchmark.py --scenarios --debug --debug-console

# Debug a specific scenario
python benchmark.py --scenario ramp_up_test --debug
```

## Debug Output

When debug mode is enabled, detailed logs are written to `debug.log` in the current directory:

```
[2024-01-15 10:30:15.123] [LOG] [+0.000s] ========================================
[2024-01-15 10:30:15.124] [LOG] [+0.001s] SCENARIO START: ramp_up_test
[2024-01-15 10:30:15.125] [LOG] [+0.002s]   Mode:        ramp_up
[2024-01-15 10:30:15.125] [LOG] [+0.002s]   Concurrency: 40
[2024-01-15 10:30:15.126] [LOG] [+0.003s]   Requests:    200
[2024-01-15 10:30:15.130] [LOG] [+0.007s] MODE START: RAMP_UP
[2024-01-15 10:30:15.131] [LOG] [+0.008s]   Target Concurrency: 40
[2024-01-15 10:30:15.131] [LOG] [+0.008s]   Ramp Up Time: 30.0s
[2024-01-15 10:30:15.132] [LOG] [+0.009s]   Steps: 10
[2024-01-15 10:30:15.132] [LOG] [+0.009s]   Hold Time: 60.0s
[2024-01-15 10:30:15.135] [LOG] [+0.012s] STEP 1/10: concurrency=4, requests=20, duration=3.0s
[2024-01-15 10:30:15.140] [DEBUG] [+0.017s] REQ #1 START (concurrent: 1)
[2024-01-15 10:30:15.523] [DEBUG] [+0.400s] REQ #1 COMPLETE: 382.5ms, success=True, tokens=45
...
[2024-01-15 10:31:45.678] [LOG] [+90.555s] STEP 1/10 COMPLETE: 20 requests in 3.012s
[2024-01-15 10:31:45.680] [LOG] [+90.557s] CONCURRENCY CHANGE: 4 -> 8 (ramp_up step 2)
...
```

## Debug Log Levels

| Level | Description |
|-------|-------------|
| `LOG` | High-level scenario and mode information |
| `DEBUG` | Request-level details (start/complete) |
| `TRACE` | Fine-grained internal operations |
| `INFO` | General information messages |
| `WARN` | Warning conditions |
| `ERROR` | Error conditions |

## Debug Information Per Mode

Each execution mode logs specific information:

| Mode | Debug Information |
|------|-------------------|
| `ramp_up` | Step progression, concurrency changes, timing |
| `stepping` | Step start/complete, requests per step |
| `spike` | Pre-spike, spike, and post-spike phases |
| `constant_rate` | Target vs actual RPS, rate adjustments |
| `arrivals` | Arrival rate, delay calculations |
| `ultimate` | Stage progression, target concurrency changes |
| `duration` | Progress checkpoints, total requests sent |

## Use Cases for Debug Logging

1. **Verify Execution Modes**: Confirm that ramp-up, stepping, and spike patterns execute as expected
2. **Trace Request Lifecycle**: See exactly when each request starts and completes
3. **Diagnose Issues**: Identify slow requests, failures, and bottlenecks
4. **Validate Concurrency**: Confirm concurrency limits are being respected
5. **Performance Analysis**: Analyze timing between steps and phases
