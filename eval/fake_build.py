"""Fake build() with the same output shape as agent/build.py. No API calls; used by tests and CI.
A prompt version containing "bad" succeeds less often, to check that the eval gate catches regressions."""
import random
import zlib


def build(sketch: bytes, prompt_version: str = "v1") -> dict:
    # zlib.crc32 (not hash()) so the same sketch gives the same fake result in every run
    rnd = random.Random(zlib.crc32(sketch[:64] + prompt_version.encode()))
    p = 0.40 if "bad" in prompt_version else 0.85
    ok = rnd.random() < p
    rounds = rnd.randint(1, 2) if ok else rnd.randint(3, 4)
    return dict(html="<html></html>", score=rnd.uniform(0.75, 0.95) if ok else rnd.uniform(0.3, 0.7),
                success=ok, rounds=rounds, latency_s=rounds * rnd.uniform(6, 12),
                api_calls=rounds + (1 if rnd.random() < 0.1 else 0), tokens=rounds * 3000,
                model_used="fake", fallback_used=rnd.random() < 0.08)
