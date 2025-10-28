"""
Latency tracking for RAG pipeline components.

Tracks component-level latencies:
- Input validation
- Vector search
- LLM generation
- Output validation
- Total end-to-end

Based on ReadyTensor Week 11 monitoring best practices.
"""

import logging
import time
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """
    Component-level latency breakdown for a single query.

    All times in seconds.
    """
    # Component latencies
    input_validation: float = 0.0
    vector_search: float = 0.0
    llm_generation: float = 0.0
    output_validation: float = 0.0
    hallucination_detection: float = 0.0

    # Total latency
    total: float = 0.0

    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Metadata
    query_id: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'input_validation_seconds': self.input_validation,
            'vector_search_seconds': self.vector_search,
            'llm_generation_seconds': self.llm_generation,
            'output_validation_seconds': self.output_validation,
            'hallucination_detection_seconds': self.hallucination_detection,
            'total_seconds': self.total,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'query_id': self.query_id,
            'user_id': self.user_id,
            # Percentages
            'input_validation_percent': (self.input_validation / self.total * 100) if self.total > 0 else 0,
            'vector_search_percent': (self.vector_search / self.total * 100) if self.total > 0 else 0,
            'llm_generation_percent': (self.llm_generation / self.total * 100) if self.total > 0 else 0,
            'output_validation_percent': (self.output_validation / self.total * 100) if self.total > 0 else 0,
            'hallucination_detection_percent': (self.hallucination_detection / self.total * 100) if self.total > 0 else 0,
        }

    def get_slowest_component(self) -> tuple[str, float]:
        """
        Identify the slowest component.

        Returns:
            Tuple of (component_name, latency_seconds)
        """
        components = {
            'input_validation': self.input_validation,
            'vector_search': self.vector_search,
            'llm_generation': self.llm_generation,
            'output_validation': self.output_validation,
            'hallucination_detection': self.hallucination_detection,
        }
        slowest = max(components.items(), key=lambda x: x[1])
        return slowest


class LatencyTracker:
    """
    Tracks latencies across RAG pipeline components.

    Provides:
    - Component-level timing
    - Context managers for easy instrumentation
    - Latency breakdown analysis
    - Integration with metrics registry

    Example:
        tracker = LatencyTracker()

        # Start tracking a query
        tracker.start_query(query_id="q123", user_id="user456")

        # Time individual components
        with tracker.track_component("input_validation"):
            validate_input(query)

        with tracker.track_component("vector_search"):
            results = vector_store.search(query)

        with tracker.track_component("llm_generation"):
            response = llm.generate(query, results)

        # Get breakdown
        breakdown = tracker.get_breakdown()
        print(f"Total: {breakdown.total:.2f}s")
        print(f"LLM: {breakdown.llm_generation:.2f}s")
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize latency tracker.

        Args:
            enabled: Whether to enable latency tracking
        """
        self.enabled = enabled
        self.current_breakdown: Optional[LatencyBreakdown] = None
        self._component_start_time: Optional[float] = None

        if enabled:
            logger.info("Latency tracker initialized")

    def start_query(
        self,
        query_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Start tracking a new query.

        Args:
            query_id: Query identifier
            user_id: User identifier
        """
        if not self.enabled:
            return

        self.current_breakdown = LatencyBreakdown(
            start_time=datetime.utcnow(),
            query_id=query_id,
            user_id=user_id
        )

    def end_query(self) -> Optional[LatencyBreakdown]:
        """
        End query tracking and calculate total latency.

        Returns:
            LatencyBreakdown with all component times
        """
        if not self.enabled or not self.current_breakdown:
            return None

        self.current_breakdown.end_time = datetime.utcnow()

        # Calculate total from components
        self.current_breakdown.total = (
            self.current_breakdown.input_validation +
            self.current_breakdown.vector_search +
            self.current_breakdown.llm_generation +
            self.current_breakdown.output_validation +
            self.current_breakdown.hallucination_detection
        )

        breakdown = self.current_breakdown
        self.current_breakdown = None  # Reset for next query

        return breakdown

    @contextmanager
    def track_component(self, component_name: str):
        """
        Context manager for tracking a component's latency.

        Args:
            component_name: Name of component (input_validation, vector_search, etc.)

        Example:
            with tracker.track_component("llm_generation"):
                response = llm.generate(prompt)
        """
        if not self.enabled or not self.current_breakdown:
            yield
            return

        # Start timing
        start_time = time.time()

        try:
            yield
        finally:
            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Store in appropriate field
            if component_name == "input_validation":
                self.current_breakdown.input_validation = elapsed
            elif component_name == "vector_search":
                self.current_breakdown.vector_search = elapsed
            elif component_name == "llm_generation":
                self.current_breakdown.llm_generation = elapsed
            elif component_name == "output_validation":
                self.current_breakdown.output_validation = elapsed
            elif component_name == "hallucination_detection":
                self.current_breakdown.hallucination_detection = elapsed
            else:
                logger.warning(f"Unknown component: {component_name}")

            logger.debug(f"{component_name} took {elapsed:.3f}s")

    def get_breakdown(self) -> Optional[LatencyBreakdown]:
        """
        Get current latency breakdown (before query ends).

        Returns:
            Current LatencyBreakdown or None
        """
        return self.current_breakdown

    def record_component_time(
        self,
        component_name: str,
        latency: float
    ) -> None:
        """
        Manually record a component's latency.

        Alternative to using track_component() context manager.

        Args:
            component_name: Component name
            latency: Latency in seconds
        """
        if not self.enabled or not self.current_breakdown:
            return

        if component_name == "input_validation":
            self.current_breakdown.input_validation = latency
        elif component_name == "vector_search":
            self.current_breakdown.vector_search = latency
        elif component_name == "llm_generation":
            self.current_breakdown.llm_generation = latency
        elif component_name == "output_validation":
            self.current_breakdown.output_validation = latency
        elif component_name == "hallucination_detection":
            self.current_breakdown.hallucination_detection = latency
        else:
            logger.warning(f"Unknown component: {component_name}")
