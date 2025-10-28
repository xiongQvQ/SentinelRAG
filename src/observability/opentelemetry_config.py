"""
OpenTelemetry configuration for distributed tracing.

Provides:
- Trace context propagation
- Span creation and management
- Integration with Langfuse and Prometheus
- Custom instrumentation

Based on ReadyTensor Week 11 observability best practices.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    TracerProvider = None
    Resource = None

logger = logging.getLogger(__name__)


class OpenTelemetryConfig:
    """
    OpenTelemetry configuration and utilities.

    Provides distributed tracing for:
    - RAG pipeline components
    - Vector database operations
    - LLM API calls
    - Custom business logic

    Example:
        otel = OpenTelemetryConfig(service_name="rag-system")

        with otel.start_span("vector_search") as span:
            span.set_attribute("query_length", len(query))
            results = vector_store.search(query)
            span.set_attribute("result_count", len(results))
    """

    def __init__(
        self,
        service_name: str = "rag-system",
        service_version: str = "1.0.0",
        environment: str = "development",
        enabled: bool = True
    ):
        """
        Initialize OpenTelemetry tracing.

        Args:
            service_name: Name of the service for trace attribution
            service_version: Service version
            environment: Deployment environment (dev/staging/prod)
            enabled: Whether to enable tracing
        """
        self.enabled = enabled and OPENTELEMETRY_AVAILABLE
        self.service_name = service_name
        self.tracer: Optional[Any] = None

        if not self.enabled:
            if not OPENTELEMETRY_AVAILABLE:
                logger.warning("OpenTelemetry not installed. Tracing disabled.")
            else:
                logger.info("OpenTelemetry tracing disabled by configuration.")
            return

        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": service_name,
                "service.version": service_version,
                "deployment.environment": environment
            })

            # Set up tracer provider
            provider = TracerProvider(resource=resource)

            # Add console exporter for development
            if environment == "development":
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(
                    BatchSpanProcessor(console_exporter)
                )

            # Set as global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer instance
            self.tracer = trace.get_tracer(
                instrumenting_module_name=__name__,
                instrumenting_library_version=service_version
            )

            logger.info(f"OpenTelemetry initialized (service: {service_name}, env: {environment})")

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.enabled = False
            self.tracer = None

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[Any] = None
    ):
        """
        Start a new tracing span (context manager).

        Args:
            name: Span name (e.g., "vector_search", "llm_call")
            attributes: Initial span attributes
            kind: Span kind (INTERNAL, CLIENT, SERVER, etc.)

        Yields:
            Span object for adding attributes and events

        Example:
            with otel.start_span("retrieval", {"k": 5}) as span:
                results = retriever.get_relevant_documents(query)
                span.set_attribute("result_count", len(results))
        """
        if not self.enabled or not self.tracer:
            # No-op context manager if tracing disabled
            yield None
            return

        try:
            with self.tracer.start_as_current_span(
                name,
                kind=kind,
                attributes=attributes or {}
            ) as span:
                yield span
        except Exception as e:
            logger.error(f"Error in span '{name}': {e}")
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    def add_span_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an event to the current span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        if not self.enabled:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.add_event(name, attributes=attributes or {})
        except Exception as e:
            logger.error(f"Failed to add span event: {e}")

    def set_span_attribute(
        self,
        key: str,
        value: Any
    ) -> None:
        """
        Set an attribute on the current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute(key, value)
        except Exception as e:
            logger.error(f"Failed to set span attribute: {e}")

    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an exception in the current span.

        Args:
            exception: Exception to record
            attributes: Additional context attributes
        """
        if not self.enabled:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.record_exception(
                    exception,
                    attributes=attributes or {}
                )
                current_span.set_status(
                    Status(StatusCode.ERROR, str(exception))
                )
        except Exception as e:
            logger.error(f"Failed to record exception: {e}")
