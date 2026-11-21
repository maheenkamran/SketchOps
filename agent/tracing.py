"""Optional tracing. Uses ops/tracing.py (Langfuse, part of D2) when it exists; otherwise does nothing.
This lets the agent run on a fresh clone that doesn't have the tracing code yet."""
import uuid

try:
    from ops.tracing import end_trace, start_trace, trace_round  # noqa: F401
except ImportError:
    def start_trace(prompt_version: str) -> str:
        return uuid.uuid4().hex

    def trace_round(**kwargs):
        pass

    def end_trace(trace_id: str, output: dict):
        pass
