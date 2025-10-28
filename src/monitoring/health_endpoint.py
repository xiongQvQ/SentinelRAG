"""
FastAPI health check endpoint for RAG system.

Provides:
- /health endpoint for basic liveness
- /health/detailed for component-level status
- /metrics endpoint for Prometheus scraping

Based on ReadyTensor Week 2 monitoring best practices.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

try:
    from fastapi import FastAPI, Response
    from fastapi.responses import PlainTextResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    Response = None
    PlainTextResponse = None
    uvicorn = None

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """
    Health status for individual components.
    """

    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: Optional[str] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.status = status
        self.message = message
        self.latency_ms = latency_ms
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'latency_ms': self.latency_ms,
            'metadata': self.metadata
        }


class HealthCheckEndpoint:
    """
    Health check endpoint manager.

    Provides health status for:
    - Overall system
    - LLM API
    - Vector database
    - Guardrails
    - Observability services

    Example:
        health = HealthCheckEndpoint(
            check_llm=lambda: check_gemini_api(),
            check_vector_db=lambda: check_faiss()
        )

        app = health.create_app()
        uvicorn.run(app, host="0.0.0.0", port=8080)

        # Access endpoints:
        # GET /health - Basic liveness
        # GET /health/detailed - Component-level status
        # GET /metrics - Prometheus metrics
    """

    def __init__(
        self,
        check_llm: Optional[callable] = None,
        check_vector_db: Optional[callable] = None,
        check_guardrails: Optional[callable] = None,
        metrics_registry = None
    ):
        """
        Initialize health check endpoint.

        Args:
            check_llm: Function to check LLM API health
            check_vector_db: Function to check vector DB health
            check_guardrails: Function to check guardrails health
            metrics_registry: Prometheus metrics registry
        """
        self.check_llm = check_llm
        self.check_vector_db = check_vector_db
        self.check_guardrails = check_guardrails
        self.metrics_registry = metrics_registry

        self.start_time = datetime.utcnow()

    def _check_component(
        self,
        name: str,
        check_func: Optional[callable]
    ) -> ComponentHealth:
        """
        Check a single component's health.

        Args:
            name: Component name
            check_func: Health check function

        Returns:
            ComponentHealth status
        """
        if check_func is None:
            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                message="Not configured"
            )

        try:
            start = datetime.utcnow()
            result = check_func()
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = None
            elif isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
                message = result.get('message')
            else:
                status = HealthStatus.HEALTHY
                message = str(result)

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )

    def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dict with overall and component-level health status
        """
        components = []

        # Check LLM
        if self.check_llm:
            components.append(self._check_component("llm", self.check_llm))

        # Check vector DB
        if self.check_vector_db:
            components.append(self._check_component("vector_db", self.check_vector_db))

        # Check guardrails
        if self.check_guardrails:
            components.append(self._check_component("guardrails", self.check_guardrails))

        # Determine overall status
        unhealthy_count = sum(
            1 for c in components if c.status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for c in components if c.status == HealthStatus.DEGRADED
        )

        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            'status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': uptime_seconds,
            'components': [c.to_dict() for c in components]
        }

    def create_app(self) -> Optional[Any]:
        """
        Create FastAPI application with health endpoints.

        Returns:
            FastAPI app instance or None if FastAPI unavailable
        """
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not installed. Health endpoint unavailable.")
            return None

        app = FastAPI(
            title="RAG System Health Check",
            description="Health monitoring for RAG pipeline",
            version="1.0.0"
        )

        @app.get("/health")
        async def health():
            """Basic health check (liveness probe)."""
            return {"status": "ok"}

        @app.get("/health/detailed")
        async def health_detailed():
            """Detailed health check with component status."""
            return self.check_health()

        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            if self.metrics_registry is None:
                return PlainTextResponse("Metrics not available", status_code=503)

            try:
                metrics_text = self.metrics_registry.export_metrics()
                content_type = self.metrics_registry.get_content_type()
                return Response(content=metrics_text, media_type=content_type)
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")
                return PlainTextResponse("Metrics export failed", status_code=500)

        @app.get("/")
        async def root():
            """Root endpoint with API info."""
            return {
                "service": "RAG System Health API",
                "version": "1.0.0",
                "endpoints": {
                    "/health": "Basic liveness check",
                    "/health/detailed": "Detailed component health",
                    "/metrics": "Prometheus metrics"
                }
            }

        return app


def create_health_app(
    check_llm: Optional[callable] = None,
    check_vector_db: Optional[callable] = None,
    check_guardrails: Optional[callable] = None,
    metrics_registry = None
) -> Optional[Any]:
    """
    Convenience function to create health check app.

    Args:
        check_llm: LLM health check function
        check_vector_db: Vector DB health check function
        check_guardrails: Guardrails health check function
        metrics_registry: Prometheus metrics registry

    Returns:
        FastAPI app or None

    Example:
        app = create_health_app(
            check_llm=lambda: gemini_client.test_connection(),
            check_vector_db=lambda: vector_store.is_ready(),
            metrics_registry=metrics
        )

        if app:
            uvicorn.run(app, host="0.0.0.0", port=8080)
    """
    endpoint = HealthCheckEndpoint(
        check_llm=check_llm,
        check_vector_db=check_vector_db,
        check_guardrails=check_guardrails,
        metrics_registry=metrics_registry
    )
    return endpoint.create_app()
