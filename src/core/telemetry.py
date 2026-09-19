import time
from typing import Dict

class SystemTelemetry:
    """Exports system operational metrics for Prometheus and Grafana dashboards."""

    def __init__(self):
        self.metrics: Dict[str, float] = {
            "core_entropy": 0.0,
            "nim_latency_seconds": 0.0,
            "active_sentinels_count": 0.0,
            "onnx_throughput_samples_per_sec": 0.0
        }

    def update_entropy(self, entropy_val: float):
        self.metrics["core_entropy"] = entropy_val

    def record_nim_latency(self, duration_sec: float):
        self.metrics["nim_latency_seconds"] = duration_sec

    def export_metrics(self) -> str:
        """Generates Prometheus-formatted text metrics."""
        lines = []
        for key, value in self.metrics.items():
            lines.append(f"# TYPE sentinel_{key} gauge")
            lines.append(f"sentinel_{key} {value:.6f}")
        return "\n".join(lines)
