"""
Prometheus metrics registry for RAG system monitoring.

Collects and exposes metrics for:
- Query latency (P50, P95, P99)
- Token usage
- Cost tracking
- Error rates
- Component-level performance
- Hallucination detection scores

Based on ReadyTensor Week 11 monitoring best practices.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None
    Summary = None
    CollectorRegistry = None

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """
    Centralized metrics registry for RAG system.

    Provides Prometheus-compatible metrics:
    - Counters: Monotonically increasing values (total queries, errors)
    - Histograms: Distribution of values (latency percentiles)
    - Gauges: Current values (active requests)
    - Summaries: Statistical summaries (response times)

    Example:
        metrics = MetricsRegistry()

        # Record a query
        metrics.record_query(
            latency=1.23,
            tokens=456,
            cost=0.0012,
            hallucination_risk=0.15
        )

        # Get Prometheus metrics
        metrics_text = metrics.export_metrics()
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize metrics registry.

        Args:
            enabled: Whether to enable metrics collection
        """
        self.enabled = enabled and PROMETHEUS_AVAILABLE

        if not self.enabled:
            if not PROMETHEUS_AVAILABLE:
                logger.warning("Prometheus client not installed. Metrics disabled.")
            else:
                logger.info("Metrics collection disabled by configuration.")
            return

        # Create custom registry
        self.registry = CollectorRegistry()

        # Initialize metrics
        self._init_query_metrics()
        self._init_component_metrics()
        self._init_quality_metrics()
        self._init_resource_metrics()

        logger.info("Prometheus metrics registry initialized")

    def _init_query_metrics(self) -> None:
        """Initialize query-level metrics."""
        if not self.enabled:
            return

        # Total queries counter
        self.total_queries = Counter(
            'rag_queries_total',
            'Total number of queries processed',
            ['status'],  # success, error, rate_limited
            registry=self.registry
        )

        # Query latency histogram (in seconds)
        self.query_latency = Histogram(
            'rag_query_latency_seconds',
            'Query processing latency',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Active queries gauge
        self.active_queries = Gauge(
            'rag_active_queries',
            'Number of queries currently being processed',
            registry=self.registry
        )

    def _init_component_metrics(self) -> None:
        """Initialize component-level metrics."""
        if not self.enabled:
            return

        # Component latency histogram
        self.component_latency = Histogram(
            'rag_component_latency_seconds',
            'Component processing latency',
            ['component'],  # validation, vector_search, llm, output_validation
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )

        # Component error counter
        self.component_errors = Counter(
            'rag_component_errors_total',
            'Component error count',
            ['component', 'error_type'],
            registry=self.registry
        )

    def _init_quality_metrics(self) -> None:
        """Initialize quality and safety metrics."""
        if not self.enabled:
            return

        # Hallucination risk gauge
        self.hallucination_risk = Histogram(
            'rag_hallucination_risk',
            'Hallucination risk score (0-1)',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

        # Faithfulness score gauge
        self.faithfulness_score = Histogram(
            'rag_faithfulness_score',
            'Faithfulness score (0-1)',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )

        # Guardrails violations counter
        self.guardrails_violations = Counter(
            'rag_guardrails_violations_total',
            'Guardrails violations',
            ['validator', 'action'],  # toxic_language:block, pii:filter
            registry=self.registry
        )

    def _init_resource_metrics(self) -> None:
        """Initialize resource usage metrics."""
        if not self.enabled:
            return

        # Token usage counter
        self.tokens_used = Counter(
            'rag_tokens_total',
            'Total tokens used',
            ['type'],  # input, output
            registry=self.registry
        )

        # Cost counter (in USD)
        self.cost_total = Counter(
            'rag_cost_usd_total',
            'Total cost in USD',
            ['component'],  # llm, embeddings
            registry=self.registry
        )

        # Vector search results histogram
        self.vector_search_results = Histogram(
            'rag_vector_search_results_count',
            'Number of results from vector search',
            buckets=[0, 1, 3, 5, 10, 20, 50],
            registry=self.registry
        )

    def record_query(
        self,
        latency: float,
        status: str = "success",
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost: float = 0.0,
        hallucination_risk: Optional[float] = None,
        faithfulness_score: Optional[float] = None
    ) -> None:
        """
        Record a completed query with all metrics.

        Args:
            latency: Total query latency in seconds
            status: Query status (success, error, rate_limited)
            tokens_input: Input tokens used
            tokens_output: Output tokens used
            cost: Estimated cost in USD
            hallucination_risk: Hallucination risk score (0-1)
            faithfulness_score: Faithfulness score (0-1)
        """
        if not self.enabled:
            return

        try:
            # Record query metrics
            self.total_queries.labels(status=status).inc()
            self.query_latency.observe(latency)

            # Record token usage
            if tokens_input > 0:
                self.tokens_used.labels(type="input").inc(tokens_input)
            if tokens_output > 0:
                self.tokens_used.labels(type="output").inc(tokens_output)

            # Record cost
            if cost > 0:
                self.cost_total.labels(component="llm").inc(cost)

            # Record quality metrics
            if hallucination_risk is not None:
                self.hallucination_risk.observe(hallucination_risk)
            if faithfulness_score is not None:
                self.faithfulness_score.observe(faithfulness_score)

        except Exception as e:
            logger.error(f"Failed to record query metrics: {e}")

    def record_component_latency(
        self,
        component: str,
        latency: float
    ) -> None:
        """
        Record component latency.

        Args:
            component: Component name (validation, vector_search, llm, etc.)
            latency: Latency in seconds
        """
        if not self.enabled:
            return

        try:
            self.component_latency.labels(component=component).observe(latency)
        except Exception as e:
            logger.error(f"Failed to record component latency: {e}")

    def record_component_error(
        self,
        component: str,
        error_type: str
    ) -> None:
        """
        Record a component error.

        Args:
            component: Component name
            error_type: Error type (timeout, api_error, validation_error, etc.)
        """
        if not self.enabled:
            return

        try:
            self.component_errors.labels(
                component=component,
                error_type=error_type
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record component error: {e}")

    def record_guardrails_violation(
        self,
        validator: str,
        action: str
    ) -> None:
        """
        Record a guardrails violation.

        Args:
            validator: Validator name (toxic_language, detect_pii, etc.)
            action: Action taken (block, filter, warn)
        """
        if not self.enabled:
            return

        try:
            self.guardrails_violations.labels(
                validator=validator,
                action=action
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record guardrails violation: {e}")

    def increment_active_queries(self) -> None:
        """Increment active queries counter."""
        if not self.enabled:
            return

        try:
            self.active_queries.inc()
        except Exception as e:
            logger.error(f"Failed to increment active queries: {e}")

    def decrement_active_queries(self) -> None:
        """Decrement active queries counter."""
        if not self.enabled:
            return

        try:
            self.active_queries.dec()
        except Exception as e:
            logger.error(f"Failed to decrement active queries: {e}")

    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics text
        """
        if not self.enabled or not self.registry:
            return b""

        try:
            return generate_latest(self.registry)
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return b""

    def get_content_type(self) -> str:
        """
        Get Prometheus metrics content type.

        Returns:
            Content-Type header value for Prometheus metrics
        """
        if not self.enabled:
            return "text/plain"

        return CONTENT_TYPE_LATEST
