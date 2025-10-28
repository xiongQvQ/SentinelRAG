"""
Observability module for RAG system monitoring and tracing.

This module provides:
- LLM tracing with Langfuse
- Distributed tracing with OpenTelemetry
- Metrics collection with Prometheus
- Hallucination detection
- Latency tracking
- Cost calculation
"""

from typing import Optional

# Graceful degradation for optional dependencies
try:
    from .langfuse_tracer import LangfuseTracer
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    LangfuseTracer = None

try:
    from .opentelemetry_config import OpenTelemetryConfig
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    OpenTelemetryConfig = None

try:
    from .metrics_registry import MetricsRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    MetricsRegistry = None

try:
    from .hallucination_detector import HallucinationDetector
    HALLUCINATION_DETECTION_AVAILABLE = True
except ImportError:
    HALLUCINATION_DETECTION_AVAILABLE = False
    HallucinationDetector = None

try:
    from .latency_tracker import LatencyTracker, LatencyBreakdown
    LATENCY_TRACKING_AVAILABLE = True
except ImportError:
    LATENCY_TRACKING_AVAILABLE = False
    LatencyTracker = None
    LatencyBreakdown = None

try:
    from .cost_calculator import CostCalculator, CostBreakdown
    COST_CALCULATION_AVAILABLE = True
except ImportError:
    COST_CALCULATION_AVAILABLE = False
    CostCalculator = None
    CostBreakdown = None

__all__ = [
    'LangfuseTracer',
    'OpenTelemetryConfig',
    'MetricsRegistry',
    'HallucinationDetector',
    'LatencyTracker',
    'LatencyBreakdown',
    'CostCalculator',
    'CostBreakdown',
    'LANGFUSE_AVAILABLE',
    'OPENTELEMETRY_AVAILABLE',
    'PROMETHEUS_AVAILABLE',
    'HALLUCINATION_DETECTION_AVAILABLE',
    'LATENCY_TRACKING_AVAILABLE',
    'COST_CALCULATION_AVAILABLE',
]
