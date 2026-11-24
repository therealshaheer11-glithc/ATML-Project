import os
import time


class Deadline:
    """Wall-clock guard for Kaggle's 12 h limit. Every long loop checks expired() between shards,
    so stopping early always leaves complete, resumable shards on disk."""

    def __init__(self, hours: float, start: float | None = None):
        self.start = start if start is not None else float(os.environ.get("CB_SESSION_START", time.time()))
        self.end = self.start + hours * 3600

    def remaining_h(self) -> float:
        return (self.end - time.time()) / 3600

    def expired(self, margin_min: float = 0.0) -> bool:
        return time.time() > self.end - margin_min * 60

    def child(self, hours: float) -> "Deadline":
        """A deadline that ends `hours` from now, but never after this one."""
        d = Deadline(0, time.time())
        d.end = min(self.end, time.time() + hours * 3600)
        return d
