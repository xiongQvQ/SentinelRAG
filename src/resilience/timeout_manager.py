"""
Timeout management for RAG pipeline operations.

Provides:
- Configurable timeouts per component
- Async timeout support
- Timeout context managers

Based on ReadyTensor Week 2 resilience patterns.
"""

import logging
import asyncio
from typing import Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import signal

logger = logging.getLogger(__name__)


@dataclass
class TimeoutConfig:
    """
    Timeout configuration for different components.

    All times in seconds.
    """
    input_validation: float = 5.0
    vector_search: float = 10.0
    llm_generation: float = 30.0
    output_validation: float = 5.0
    total_query: float = 60.0


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


class TimeoutManager:
    """
    Manages timeouts for RAG pipeline operations.

    Provides timeout enforcement for:
    - Individual components
    - End-to-end queries
    - Async operations

    Example:
        timeout_mgr = TimeoutManager()

        # Sync operation
        with timeout_mgr.timeout_context(30.0, "llm_call"):
            response = llm.generate(prompt)

        # Async operation
        async def async_task():
            result = await timeout_mgr.with_async_timeout(
                async_operation(),
                timeout=10.0
            )
            return result
    """

    def __init__(
        self,
        config: Optional[TimeoutConfig] = None,
        enabled: bool = True
    ):
        """
        Initialize timeout manager.

        Args:
            config: Timeout configuration
            enabled: Whether to enforce timeouts
        """
        self.config = config or TimeoutConfig()
        self.enabled = enabled

        if enabled:
            logger.info("Timeout manager initialized")
        else:
            logger.info("Timeout enforcement disabled")

    @contextmanager
    def timeout_context(
        self,
        timeout: float,
        operation_name: str = "operation"
    ):
        """
        Context manager for sync timeout enforcement.

        Uses signal.alarm on Unix systems.

        Args:
            timeout: Timeout in seconds
            operation_name: Operation name for logging

        Raises:
            TimeoutError: If operation exceeds timeout

        Example:
            with timeout_mgr.timeout_context(30.0, "vector_search"):
                results = vector_store.search(query)
        """
        if not self.enabled:
            yield
            return

        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"{operation_name} exceeded timeout of {timeout}s"
            )

        # Set alarm (Unix only)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

            try:
                yield
            finally:
                # Cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except AttributeError:
            # signal.alarm not available (Windows)
            logger.warning(
                "signal.alarm not available. Timeout enforcement disabled."
            )
            yield

    async def with_async_timeout(
        self,
        coroutine,
        timeout: float,
        operation_name: str = "async_operation"
    ) -> Any:
        """
        Async timeout wrapper.

        Args:
            coroutine: Coroutine to execute
            timeout: Timeout in seconds
            operation_name: Operation name for logging

        Returns:
            Result of coroutine

        Raises:
            TimeoutError: If operation exceeds timeout

        Example:
            result = await timeout_mgr.with_async_timeout(
                fetch_data(),
                timeout=10.0,
                operation_name="api_call"
            )
        """
        if not self.enabled:
            return await coroutine

        try:
            return await asyncio.wait_for(coroutine, timeout=timeout)
        except asyncio.TimeoutError as e:
            msg = f"{operation_name} exceeded timeout of {timeout}s"
            logger.error(msg)
            raise TimeoutError(msg) from e

    def get_timeout_for_component(self, component: str) -> float:
        """
        Get configured timeout for a component.

        Args:
            component: Component name (input_validation, vector_search, etc.)

        Returns:
            Timeout in seconds
        """
        timeouts = {
            'input_validation': self.config.input_validation,
            'vector_search': self.config.vector_search,
            'llm_generation': self.config.llm_generation,
            'output_validation': self.config.output_validation,
            'total_query': self.config.total_query,
        }
        return timeouts.get(component, self.config.total_query)


# Convenience function
def with_timeout(timeout: float, operation_name: str = "operation"):
    """
    Decorator for adding timeout to a function.

    Args:
        timeout: Timeout in seconds
        operation_name: Operation name for logging

    Returns:
        Decorated function

    Example:
        @with_timeout(30.0, "llm_call")
        def call_llm(prompt):
            return llm.generate(prompt)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            timeout_mgr = TimeoutManager()
            with timeout_mgr.timeout_context(timeout, operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
