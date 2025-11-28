"""
Export handlers for benchmark results.
Supports Markdown and CSV formats.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import BenchmarkMetrics


class BaseExporter:
    """Base class for exporters."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(self, metrics: List[BenchmarkMetrics], filename: Optional[str] = None) -> str:
        """Export metrics to file."""
        raise NotImplementedError
    
    def get_extension(self) -> str:
        """Get file extension for this exporter."""
        raise NotImplementedError
    
    def generate_filename(self, prefix: str = "benchmark") -> str:
        """Generate a unique filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{self.get_extension()}"


class MarkdownExporter(BaseExporter):
    """Export benchmark results to Markdown format."""
    
    def get_extension(self) -> str:
        return "md"
    
    def export(self, metrics: List[BenchmarkMetrics], filename: Optional[str] = None) -> str:
        """Export metrics to Markdown file."""
        if not filename:
            filename = self.generate_filename()
        
        filepath = self.output_dir / filename
        
        content = self._generate_markdown(metrics)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(filepath)
    
    def _generate_markdown(self, metrics: List[BenchmarkMetrics]) -> str:
        """Generate Markdown content from metrics."""
        lines = []
        
        # Header
        lines.append("# LLM Benchmark Results")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Scenario | Model | Type | Requests | Concurrency | Success Rate | Duration | Req/s | Tokens/s |")
        lines.append("|----------|-------|------|----------|-------------|--------------|----------|-------|----------|")
        
        for m in metrics:
            success_rate = (m.successful_requests / m.total_requests * 100) if m.total_requests > 0 else 0
            tokens_per_sec = f"{m.tokens_per_sec:.2f}" if m.tokens_per_sec else "N/A"
            lines.append(
                f"| {m.scenario_name} | {m.model_name} | {m.model_type} | "
                f"{m.total_requests} | {m.concurrency} | {success_rate:.1f}% | "
                f"{m.duration:.2f}s | {m.requests_per_sec:.2f} | {tokens_per_sec} |"
            )
        
        lines.append("")
        
        # Detailed results for each benchmark
        lines.append("## Detailed Results")
        lines.append("")
        
        for i, m in enumerate(metrics, 1):
            lines.append(f"### {i}. {m.scenario_name}")
            lines.append("")
            lines.append(f"**Model:** {m.model_name} ({m.model_type})")
            lines.append("")
            
            # Configuration
            lines.append("#### Configuration")
            lines.append("")
            lines.append(f"- Total Requests: {m.total_requests}")
            lines.append(f"- Concurrency Level: {m.concurrency}")
            lines.append("")
            
            # Results
            lines.append("#### Results")
            lines.append("")
            lines.append(f"- ✅ Successful Requests: {m.successful_requests}")
            lines.append(f"- ❌ Failed Requests: {m.failed_requests}")
            lines.append(f"- ⏱️ Duration: {m.duration:.3f}s")
            lines.append(f"- 📊 Requests/sec: {m.requests_per_sec:.2f}")
            if m.tokens_per_sec:
                lines.append(f"- 🔤 Tokens/sec: {m.tokens_per_sec:.2f}")
                lines.append(f"- ⚡ ns/Token: {m.ns_per_token:.2f}")
            lines.append("")
            
            # Latency
            lines.append("#### Latency Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Average | {m.avg_latency:.4f}s |")
            lines.append(f"| P50 (Median) | {m.p50_latency:.4f}s |")
            lines.append(f"| P95 | {m.p95_latency:.4f}s |")
            lines.append(f"| P99 | {m.p99_latency:.4f}s |")
            lines.append(f"| Min | {m.min_latency:.4f}s |")
            lines.append(f"| Max | {m.max_latency:.4f}s |")
            lines.append("")
            
            # Token metrics
            if m.total_tokens > 0:
                lines.append("#### Token Metrics")
                lines.append("")
                lines.append(f"- Total Tokens: {m.total_tokens}")
                lines.append(f"- Prompt Tokens: {m.total_prompt_tokens}")
                lines.append(f"- Completion Tokens: {m.total_completion_tokens}")
                lines.append(f"- Avg Tokens/Request: {m.avg_tokens_per_request:.2f}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)


class CSVExporter(BaseExporter):
    """Export benchmark results to CSV format."""
    
    def get_extension(self) -> str:
        return "csv"
    
    def export(self, metrics: List[BenchmarkMetrics], filename: Optional[str] = None) -> str:
        """Export metrics to CSV file."""
        if not filename:
            filename = self.generate_filename()
        
        filepath = self.output_dir / filename
        
        self._write_csv(metrics, filepath)
        
        return str(filepath)
    
    def _write_csv(self, metrics: List[BenchmarkMetrics], filepath: Path) -> None:
        """Write metrics to CSV file."""
        headers = [
            "scenario_name",
            "model_name",
            "model_type",
            "timestamp",
            "total_requests",
            "concurrency",
            "successful_requests",
            "failed_requests",
            "duration_seconds",
            "requests_per_sec",
            "tokens_per_sec",
            "ns_per_token",
            "avg_latency_sec",
            "p50_latency_sec",
            "p95_latency_sec",
            "p99_latency_sec",
            "min_latency_sec",
            "max_latency_sec",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "avg_tokens_per_request"
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for m in metrics:
                row = [
                    m.scenario_name,
                    m.model_name,
                    m.model_type,
                    m.start_time.isoformat() if m.start_time else "",
                    m.total_requests,
                    m.concurrency,
                    m.successful_requests,
                    m.failed_requests,
                    round(m.duration, 3),
                    round(m.requests_per_sec, 2),
                    round(m.tokens_per_sec, 2) if m.tokens_per_sec else "",
                    round(m.ns_per_token, 2) if m.ns_per_token else "",
                    round(m.avg_latency, 4),
                    round(m.p50_latency, 4),
                    round(m.p95_latency, 4),
                    round(m.p99_latency, 4),
                    round(m.min_latency, 4),
                    round(m.max_latency, 4),
                    m.total_tokens,
                    m.total_prompt_tokens,
                    m.total_completion_tokens,
                    round(m.avg_tokens_per_request, 2)
                ]
                writer.writerow(row)


class JSONExporter(BaseExporter):
    """Export benchmark results to JSON format."""
    
    def get_extension(self) -> str:
        return "json"
    
    def export(self, metrics: List[BenchmarkMetrics], filename: Optional[str] = None) -> str:
        """Export metrics to JSON file."""
        if not filename:
            filename = self.generate_filename()
        
        filepath = self.output_dir / filename
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "benchmarks": [m.to_dict() for m in metrics]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)


def get_exporter(format_type: str, output_dir: str = "results") -> BaseExporter:
    """Factory function to get the appropriate exporter."""
    exporters = {
        "markdown": MarkdownExporter,
        "md": MarkdownExporter,
        "csv": CSVExporter,
        "json": JSONExporter
    }
    
    format_type = format_type.lower()
    exporter_class = exporters.get(format_type)
    
    if exporter_class is None:
        raise ValueError(f"Unknown export format: {format_type}. Supported formats: {list(exporters.keys())}")
    
    return exporter_class(output_dir)


def export_results(
    metrics: List[BenchmarkMetrics],
    formats: List[str],
    output_dir: str = "results"
) -> Dict[str, str]:
    """Export results to multiple formats."""
    results = {}
    
    for fmt in formats:
        exporter = get_exporter(fmt, output_dir)
        filepath = exporter.export(metrics)
        results[fmt] = filepath
    
    return results
