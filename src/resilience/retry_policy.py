"""
Retry policies with exponential backoff using Tenacity.

Provides:
- Configurable retry policies
- Exponential backoff
- Component-specific retry strategies
- Integration with logging

Based on ReadyTensor Week 2 resilience patterns.
"""

import logging
from typing import Callable, Optional, Any
from functools import wraps

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
        RetryError
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    retry = None
    stop_after_attempt = None
    wait_exponential = None
    retry_if_exception_type = None
    before_sleep_log = None
    RetryError = None

logger = logging.getLogger(__name__)


class RetryPolicy:
    """
    Configurable retry policy for different components.

    Provides pre-configured retry strategies for:
    - LLM API calls
    - Vector database operations
    - Generic API calls

    Example:
        policy = RetryPolicy()

        @policy.llm_retry()
        def call_llm(prompt):
            return gemini_client.generate(prompt)

        response = call_llm("What is AI?")
    """

    def __init__(
        self,
        enabled: bool = True,
        max_attempts: int = 3,
        min_wait: int = 1,
        max_wait: int = 10,
        exponential_multiplier: int = 2
    ):
        """
        Initialize retry policy.

        Args:
            enabled: Whether to enable retries
            max_attempts: Maximum number of retry attempts
            min_wait: Minimum wait time in seconds
            max_wait: Maximum wait time in seconds
            exponential_multiplier: Multiplier for exponential backoff
        """
        self.enabled = enabled and TENACITY_AVAILABLE
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.exponential_multiplier = exponential_multiplier

        if not self.enabled:
            if not TENACITY_AVAILABLE:
                logger.warning("Tenacity not installed. Retry logic disabled.")
            else:
                logger.info("Retry logic disabled by configuration.")

    def llm_retry(
        self,
        max_attempts: Optional[int] = None,
        min_wait: Optional[int] = None,
        max_wait: Optional[int] = None
    ):
        """
        Retry decorator for LLM API calls.

        Retries on common LLM errors:
        - Rate limit errors
        - Temporary API failures
        - Timeout errors

        Args:
            max_attempts: Override default max attempts
            min_wait: Override default min wait
            max_wait: Override default max wait

        Returns:
            Decorator function

        Example:
            @retry_policy.llm_retry()
            def call_gemini(prompt):
                return gemini.generate(prompt)
        """
        if not self.enabled:
            # No-op decorator if disabled
            def decorator(func):
                return func
            return decorator

        return retry(
            stop=stop_after_attempt(max_attempts or self.max_attempts),
            wait=wait_exponential(
                multiplier=self.exponential_multiplier,
                min=min_wait or self.min_wait,
                max=max_wait or self.max_wait
            ),
            retry=retry_if_exception_type((
                Exception,  # Catch all for now
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )

    def vector_search_retry(
        self,
        max_attempts: Optional[int] = None
    ):
        """
        Retry decorator for vector database operations.

        Typically fewer retries than LLM calls since vector
        operations are usually faster to fail permanently.

        Args:
            max_attempts: Override default (typically 2 for vector ops)

        Returns:
            Decorator function

        Example:
            @retry_policy.vector_search_retry()
            def search_vectors(query):
                return vector_store.similarity_search(query)
        """
        if not self.enabled:
            def decorator(func):
                return func
            return decorator

        return retry(
            stop=stop_after_attempt(max_attempts or 2),
            wait=wait_exponential(
                multiplier=1,
                min=1,
                max=5
            ),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )

    def api_call_retry(
        self,
        max_attempts: Optional[int] = None
    ):
        """
        Generic retry decorator for API calls.

        Args:
            max_attempts: Override default max attempts

        Returns:
            Decorator function

        Example:
            @retry_policy.api_call_retry()
            def fetch_data():
                return requests.get(url).json()
        """
        if not self.enabled:
            def decorator(func):
                return func
            return decorator

        return retry(
            stop=stop_after_attempt(max_attempts or self.max_attempts),
            wait=wait_exponential(
                multiplier=self.exponential_multiplier,
                min=self.min_wait,
                max=self.max_wait
            ),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )


# Module-level convenience functions
_default_policy = RetryPolicy()


def retry_with_policy(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
    exponential_multiplier: int = 2
):
    """
    Custom retry decorator with specific parameters.

    Args:
        max_attempts: Maximum retry attempts
        min_wait: Minimum wait time
        max_wait: Maximum wait time
        exponential_multiplier: Backoff multiplier

    Returns:
        Decorator function

    Example:
        @retry_with_policy(max_attempts=5, min_wait=2, max_wait=30)
        def critical_operation():
            return perform_operation()
    """
    if not TENACITY_AVAILABLE:
        def decorator(func):
            return func
        return decorator

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=exponential_multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


# Convenience decorators using default policy
def llm_retry():
    """Shorthand for LLM retry decorator."""
    return _default_policy.llm_retry()


def vector_search_retry():
    """Shorthand for vector search retry decorator."""
    return _default_policy.vector_search_retry()


def api_call_retry():
    """Shorthand for API call retry decorator."""
    return _default_policy.api_call_retry()
