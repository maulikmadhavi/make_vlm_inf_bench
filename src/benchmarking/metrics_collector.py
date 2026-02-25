"""
Collects GPU utilization metrics by polling pynvml in a background thread
while inference requests are in-flight.
"""
import threading
import time
from dataclasses import dataclass, field

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

from src.utils.config import GPU_POLL_INTERVAL
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GPUSample:
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float

    @property
    def memory_pct(self) -> float:
        return (self.memory_used_mb / self.memory_total_mb * 100) if self.memory_total_mb else 0.0


class GPUMonitor:
    """Polls GPU stats in a background thread during inference."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._samples: list[GPUSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as e:
                log.warning(f"pynvml init failed: {e}. GPU metrics will be unavailable.")
        else:
            log.warning("pynvml not installed. GPU metrics will be unavailable.")

    def start(self) -> None:
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def close(self) -> None:
        """Release pynvml resources."""
        if getattr(self, "_stop_event", None) is not None:
            self.stop()
        if _PYNVML_AVAILABLE and getattr(self, "_handle", None) is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            if self._handle:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._samples.append(GPUSample(
                        memory_used_mb=mem.used / 1024**2,
                        memory_total_mb=mem.total / 1024**2,
                        utilization_pct=util.gpu,
                    ))
                except Exception:
                    pass
            time.sleep(GPU_POLL_INTERVAL)

    def summary(self) -> dict:
        if not self._samples:
            return {"gpu_available": False}
        mem_pcts = [s.memory_pct for s in self._samples]
        util_pcts = [s.utilization_pct for s in self._samples]
        used_mbs = [s.memory_used_mb for s in self._samples]
        # memory_total_mb is constant for a given device; any sample is representative
        total_mb = self._samples[0].memory_total_mb
        return {
            "gpu_available": True,
            "memory_utilization_pct": round(sum(mem_pcts) / len(mem_pcts), 1),
            "compute_utilization_pct": round(sum(util_pcts) / len(util_pcts), 1),
            "memory_used_mb": round(sum(used_mbs) / len(used_mbs), 1),
            "memory_total_mb": round(total_mb, 1),
        }


def aggregate(values: list[float]) -> dict:
    """Return mean, std, min, max, p50, p95, p99 for a list of floats."""
    import statistics

    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        # Linear interpolation (consistent with numpy percentile / statistics.quantiles)
        pos = (p / 100) * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    return {
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.stdev(values) if n > 1 else 0.0, 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "p50": round(percentile(50), 3),
        "p95": round(percentile(95), 3),
        "p99": round(percentile(99), 3),
    }
