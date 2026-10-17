"""Read-only ROCm health monitoring. Never writes hardware controls."""
import threading
import time
import psutil
import math
import re
from red_curtain.gpu import read_gpu_snapshot


def gpu_snapshot():
    return read_gpu_snapshot(0)


def check_snapshot(snapshot):
    if not snapshot:
        raise RuntimeError("No GPU health readings available")
    limits = {"edge": 80, "junction": 95, "memory": 90}
    for card in snapshot.values():
        usable = 0
        for key, value in card.items():
            match=re.fullmatch(r'Temperature \(Sensor (.+)\) \(C\)',key)
            if not match or value is None or str(value) in {'N/A','None',''}:
                continue
            sensor=match[1]
            try:temperature=float(value)
            except (ValueError,TypeError):raise RuntimeError(f'GPU temperature sensor invalid: {sensor}={value}')
            if not math.isfinite(temperature) or not -40<=temperature<=200:
                raise RuntimeError(f"GPU temperature sensor invalid: {sensor}={value}")
            usable += 1
            limit=limits.get(sensor,80)
            critical=card.get(f'Temperature critical (Sensor {sensor}) (C)')
            if critical is not None:
                try:critical=float(critical)
                except (ValueError,TypeError):raise RuntimeError(f'GPU critical temperature reading invalid: {sensor}')
                if not math.isfinite(critical) or critical<=5:
                    raise RuntimeError(f'GPU critical temperature reading invalid: {sensor}')
                limit=min(limit,critical-5)
            if temperature >= limit:
                raise RuntimeError(f"GPU {sensor} {value} °C exceeds test limit {limit} °C")
        if not usable:
            raise RuntimeError('No usable GPU temperature sensor is available; refusing unmonitored inference')


class Monitor:
    def __init__(self, interval=2):
        self.interval = interval
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def _sample(self):
        try:
            sample = gpu_snapshot()
            self.samples.append({"time": time.time(), "gpu": sample,
                                 "process_rss_bytes": psutil.Process().memory_info().rss})
            check_snapshot(sample)
        except Exception as exc:
            self.error = f"GPU safety monitor stopped inference: {exc}"

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self._sample()
            if self.error:
                return

    def __enter__(self):
        self._sample()
        self.check()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=12)
        self._sample()
