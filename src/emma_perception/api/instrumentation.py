from opentelemetry import trace

from emma_perception._version import __version__  # noqa: WPS436


def get_tracer(name: str) -> trace.Tracer:
    """Get OTEL tracer."""
    return trace.get_tracer(name, __version__)
