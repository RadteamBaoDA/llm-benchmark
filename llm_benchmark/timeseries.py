"""
Timeseries metrics recording for LLM benchmarks.
Records request-level metrics with timestamps for detailed analysis.
"""

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class TimeseriesRecord:
    """A single timeseries record for a request."""
    timestamp: float  # Unix timestamp in seconds
    elapsed_ms: float  # Time since benchmark start in milliseconds
    scenario_name: str
    model_name: str
    model_type: str
    request_id: int
    latency_ms: float  # Request latency in milliseconds
    success: bool
    status_code: Optional[int] = None
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    concurrent_requests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TimeseriesWriter:
    """Writes timeseries metrics to files during benchmark execution."""
    
    CSV_HEADERS = [
        "timestamp", "elapsed_ms", "scenario_name", "model_name", "model_type",
        "request_id", "latency_ms", "success", "status_code", "tokens",
        "prompt_tokens", "completion_tokens", "error", "concurrent_requests"
    ]
    
    def __init__(self, output_dir: str = "results", format: str = "csv"):
        """
        Initialize timeseries writer.
        
        Args:
            output_dir: Directory to write timeseries files
            format: Output format ('csv' or 'jsonl')
        """
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = Lock()
        self._start_time: Optional[float] = None
        self._request_counter: int = 0
        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._csv_writer = None
        self._records: List[TimeseriesRecord] = []
    
    def start_scenario(self, scenario_name: str, model_name: str) -> str:
        """
        Start recording for a new scenario.
        
        Returns:
            Path to the timeseries file
        """
        self._close_file()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_scenario = scenario_name.replace(" ", "_").replace("/", "-")
        safe_model = model_name.replace("/", "-")
        
        if self.format == "csv":
            filename = f"timeseries_{safe_scenario}_{safe_model}_{timestamp}.csv"
        else:
            filename = f"timeseries_{safe_scenario}_{safe_model}_{timestamp}.jsonl"
        
        self._current_file = self.output_dir / filename
        self._start_time = time.time()
        self._request_counter = 0
        self._records = []
        
        # Open file and write headers
        self._file_handle = open(self._current_file, 'w', newline='', encoding='utf-8')
        
        if self.format == "csv":
            self._csv_writer = csv.DictWriter(self._file_handle, fieldnames=self.CSV_HEADERS)
            self._csv_writer.writeheader()
        
        return str(self._current_file)
    
    def record(
        self,
        scenario_name: str,
        model_name: str,
        model_type: str,
        latency: float,
        success: bool,
        status_code: Optional[int] = None,
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[str] = None,
        concurrent_requests: int = 0
    ) -> None:
        """Record a single request metric."""
        with self._lock:
            if self._start_time is None:
                self._start_time = time.time()
            
            self._request_counter += 1
            current_time = time.time()
            
            record = TimeseriesRecord(
                timestamp=current_time,
                elapsed_ms=(current_time - self._start_time) * 1000,
                scenario_name=scenario_name,
                model_name=model_name,
                model_type=model_type,
                request_id=self._request_counter,
                latency_ms=latency * 1000,  # Convert to ms
                success=success,
                status_code=status_code,
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=error,
                concurrent_requests=concurrent_requests
            )
            
            self._records.append(record)
            self._write_record(record)
    
    def _write_record(self, record: TimeseriesRecord) -> None:
        """Write a record to the file."""
        if self._file_handle is None:
            return
        
        if self.format == "csv":
            self._csv_writer.writerow(record.to_dict())
        else:
            self._file_handle.write(json.dumps(record.to_dict()) + "\n")
        
        self._file_handle.flush()
    
    def end_scenario(self) -> List[TimeseriesRecord]:
        """End recording for current scenario and return all records."""
        records = self._records.copy()
        self._close_file()
        return records
    
    def _close_file(self) -> None:
        """Close the current file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            self._csv_writer = None
    
    def get_current_file(self) -> Optional[str]:
        """Get the current timeseries file path."""
        return str(self._current_file) if self._current_file else None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_file()


class TimeseriesReader:
    """Reads timeseries data from files for analysis and reporting."""
    
    def __init__(self, file_path: str):
        """
        Initialize reader with a timeseries file.
        
        Args:
            file_path: Path to CSV or JSONL timeseries file
        """
        self.file_path = Path(file_path)
        self.records: List[TimeseriesRecord] = []
        self._load()
    
    def _load(self) -> None:
        """Load records from file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Timeseries file not found: {self.file_path}")
        
        suffix = self.file_path.suffix.lower()
        
        if suffix == ".csv":
            self._load_csv()
        elif suffix == ".jsonl":
            self._load_jsonl()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_csv(self) -> None:
        """Load records from CSV file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = TimeseriesRecord(
                    timestamp=float(row["timestamp"]),
                    elapsed_ms=float(row["elapsed_ms"]),
                    scenario_name=row["scenario_name"],
                    model_name=row["model_name"],
                    model_type=row["model_type"],
                    request_id=int(row["request_id"]),
                    latency_ms=float(row["latency_ms"]),
                    success=row["success"].lower() == "true",
                    status_code=int(row["status_code"]) if row["status_code"] else None,
                    tokens=int(row["tokens"]) if row["tokens"] else 0,
                    prompt_tokens=int(row["prompt_tokens"]) if row["prompt_tokens"] else 0,
                    completion_tokens=int(row["completion_tokens"]) if row["completion_tokens"] else 0,
                    error=row["error"] if row["error"] else None,
                    concurrent_requests=int(row["concurrent_requests"]) if row["concurrent_requests"] else 0
                )
                self.records.append(record)
    
    def _load_jsonl(self) -> None:
        """Load records from JSONL file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    record = TimeseriesRecord(**data)
                    self.records.append(record)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from timeseries data."""
        if not self.records:
            return {}
        
        successful = [r for r in self.records if r.success]
        failed = [r for r in self.records if not r.success]
        
        latencies = [r.latency_ms for r in successful]
        
        import numpy as np
        
        # Time range
        start_time = min(r.timestamp for r in self.records)
        end_time = max(r.timestamp for r in self.records)
        duration = end_time - start_time
        
        stats = {
            "scenario_name": self.records[0].scenario_name,
            "model_name": self.records[0].model_name,
            "model_type": self.records[0].model_type,
            "total_requests": len(self.records),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(self.records) * 100 if self.records else 0,
            "duration_seconds": duration,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
        }
        
        if latencies:
            stats.update({
                "latency_avg_ms": float(np.mean(latencies)),
                "latency_min_ms": float(np.min(latencies)),
                "latency_max_ms": float(np.max(latencies)),
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p90_ms": float(np.percentile(latencies, 90)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "latency_p99_ms": float(np.percentile(latencies, 99)),
                "latency_std_ms": float(np.std(latencies)),
            })
            
            # Throughput
            if duration > 0:
                stats["requests_per_sec"] = len(successful) / duration
                
                total_tokens = sum(r.tokens for r in successful)
                if total_tokens > 0:
                    stats["tokens_per_sec"] = total_tokens / duration
                    stats["total_tokens"] = total_tokens
        
        # Error breakdown
        if failed:
            error_counts = {}
            for r in failed:
                err = r.error or "Unknown error"
                error_counts[err] = error_counts.get(err, 0) + 1
            stats["errors"] = error_counts
        
        return stats
    
    def get_latency_over_time(self, bucket_size_ms: int = 1000) -> List[Dict[str, Any]]:
        """
        Get latency metrics bucketed by time.
        
        Args:
            bucket_size_ms: Size of time buckets in milliseconds
            
        Returns:
            List of buckets with latency statistics
        """
        if not self.records:
            return []
        
        import numpy as np
        
        # Group by time buckets
        min_elapsed = min(r.elapsed_ms for r in self.records)
        max_elapsed = max(r.elapsed_ms for r in self.records)
        
        buckets = []
        current_bucket = min_elapsed
        
        while current_bucket <= max_elapsed:
            bucket_records = [
                r for r in self.records
                if current_bucket <= r.elapsed_ms < current_bucket + bucket_size_ms
                and r.success
            ]
            
            if bucket_records:
                latencies = [r.latency_ms for r in bucket_records]
                buckets.append({
                    "elapsed_ms": current_bucket + bucket_size_ms / 2,
                    "count": len(bucket_records),
                    "avg_latency_ms": float(np.mean(latencies)),
                    "min_latency_ms": float(np.min(latencies)),
                    "max_latency_ms": float(np.max(latencies)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)) if len(latencies) > 1 else latencies[0],
                })
            
            current_bucket += bucket_size_ms
        
        return buckets
    
    def get_throughput_over_time(self, bucket_size_ms: int = 1000) -> List[Dict[str, Any]]:
        """
        Get throughput metrics bucketed by time.
        
        Args:
            bucket_size_ms: Size of time buckets in milliseconds
            
        Returns:
            List of buckets with throughput statistics
        """
        if not self.records:
            return []
        
        min_elapsed = min(r.elapsed_ms for r in self.records)
        max_elapsed = max(r.elapsed_ms for r in self.records)
        
        buckets = []
        current_bucket = min_elapsed
        
        while current_bucket <= max_elapsed:
            bucket_records = [
                r for r in self.records
                if current_bucket <= r.elapsed_ms < current_bucket + bucket_size_ms
            ]
            
            if bucket_records:
                successful = [r for r in bucket_records if r.success]
                failed = [r for r in bucket_records if not r.success]
                tokens = sum(r.tokens for r in successful)
                
                buckets.append({
                    "elapsed_ms": current_bucket + bucket_size_ms / 2,
                    "requests": len(bucket_records),
                    "successful": len(successful),
                    "failed": len(failed),
                    "tokens": tokens,
                    "requests_per_sec": len(successful) / (bucket_size_ms / 1000),
                    "tokens_per_sec": tokens / (bucket_size_ms / 1000) if tokens > 0 else 0,
                })
            
            current_bucket += bucket_size_ms
        
        return buckets
    
    def get_latency_distribution(self, bins: int = 50) -> Dict[str, Any]:
        """
        Get latency distribution for histogram.
        
        Args:
            bins: Number of histogram bins
            
        Returns:
            Dictionary with histogram data
        """
        if not self.records:
            return {"bins": [], "counts": [], "edges": []}
        
        import numpy as np
        
        latencies = [r.latency_ms for r in self.records if r.success]
        if not latencies:
            return {"bins": [], "counts": [], "edges": []}
        
        counts, edges = np.histogram(latencies, bins=bins)
        
        return {
            "bins": [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))],
            "counts": counts.tolist(),
            "edges": edges.tolist()
        }


def load_all_timeseries(directory: str) -> List[TimeseriesReader]:
    """
    Load all timeseries files from a directory.
    
    Args:
        directory: Path to directory containing timeseries files
        
    Returns:
        List of TimeseriesReader instances
    """
    dir_path = Path(directory)
    readers = []
    
    for file in dir_path.glob("timeseries_*.csv"):
        try:
            readers.append(TimeseriesReader(str(file)))
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    for file in dir_path.glob("timeseries_*.jsonl"):
        try:
            readers.append(TimeseriesReader(str(file)))
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    return readers
