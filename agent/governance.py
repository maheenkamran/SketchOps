"""Loop governance: round budget, early stop on no improvement, per-build timeout."""
from typing import List, Optional


class Governance:
    def __init__(self, max_rounds: int, min_improvement: float, timeout_s: float):
        self.max_rounds = max_rounds
        self.min_improvement = min_improvement
        self.timeout_s = timeout_s

    def should_stop(self, rounds_done: int, scores: List[float], elapsed_s: float) -> Optional[str]:
        """Return a stop reason, or None to keep going."""
        if elapsed_s >= self.timeout_s:
            return "timeout"
        if rounds_done >= self.max_rounds:
            return "max_rounds"
        if len(scores) >= 2 and scores[-1] - max(scores[:-1]) < self.min_improvement:
            return "no_improvement"
        return None
