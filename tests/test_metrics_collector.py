"""Tests for metrics_collector: aggregate() and GPUMonitor.summary()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking.metrics_collector import GPUSample, GPUMonitor, aggregate


class TestAggregate:
    def test_empty_returns_empty_dict(self):
        assert aggregate([]) == {}

    def test_single_value(self):
        result = aggregate([42.0])
        assert result["mean"] == 42.0
        assert result["min"] == 42.0
        assert result["max"] == 42.0
        assert result["std"] == 0.0
        assert result["p50"] == 42.0
        assert result["p95"] == 42.0
        assert result["p99"] == 42.0

    def test_percentiles_correct(self):
        # 11 evenly-spaced values: 0..10
        values = list(range(11))
        result = aggregate(values)
        assert result["p50"] == 5.0
        # p95 of [0..10]: position = 0.95*10 = 9.5 → interpolated between 9 and 10 = 9.5
        assert result["p95"] == 9.5
        # p99 of [0..10]: position = 0.99*10 = 9.9 → interpolated between 9 and 10 = 9.9
        assert abs(result["p99"] - 9.9) < 1e-6

    def test_mean_and_std(self):
        import statistics
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = aggregate(values)
        assert result["mean"] == round(statistics.mean(values), 3)
        assert result["std"] == round(statistics.stdev(values), 3)


class TestGPUMonitorSummary:
    def _make_monitor_with_samples(self, samples: list[GPUSample]) -> GPUMonitor:
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor._samples = samples
        monitor._stop_event = None
        monitor._thread = None
        monitor._handle = None
        monitor._device_index = 0
        return monitor

    def test_no_samples_returns_unavailable(self):
        monitor = self._make_monitor_with_samples([])
        result = monitor.summary()
        assert result == {"gpu_available": False}

    def test_memory_used_is_mean_not_last(self):
        samples = [
            GPUSample(memory_used_mb=100.0, memory_total_mb=1000.0, utilization_pct=50.0),
            GPUSample(memory_used_mb=200.0, memory_total_mb=1000.0, utilization_pct=60.0),
            GPUSample(memory_used_mb=300.0, memory_total_mb=1000.0, utilization_pct=70.0),
        ]
        monitor = self._make_monitor_with_samples(samples)
        result = monitor.summary()
        assert result["gpu_available"] is True
        # mean of 100, 200, 300 = 200; NOT 300 (the last sample)
        assert result["memory_used_mb"] == 200.0
        assert result["memory_total_mb"] == 1000.0
        assert result["compute_utilization_pct"] == 60.0

    def test_memory_utilization_pct_is_mean(self):
        samples = [
            GPUSample(memory_used_mb=500.0, memory_total_mb=1000.0, utilization_pct=30.0),
            GPUSample(memory_used_mb=700.0, memory_total_mb=1000.0, utilization_pct=70.0),
        ]
        monitor = self._make_monitor_with_samples(samples)
        result = monitor.summary()
        # memory_pct: 50.0 and 70.0 → mean = 60.0
        assert result["memory_utilization_pct"] == 60.0
