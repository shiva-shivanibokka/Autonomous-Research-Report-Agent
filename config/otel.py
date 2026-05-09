"""OpenTelemetry tracing setup — exports to Jaeger via OTLP."""

import os


def setup_tracing(service_name: str = "research-agent"):
    """Configure OpenTelemetry with OTLP exporter pointing to Jaeger."""
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        otlp_endpoint = os.environ.get("OTLP_ENDPOINT", "http://jaeger:4317")

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    except ImportError:
        # If OTel packages not installed, skip silently
        pass
