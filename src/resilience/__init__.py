"""
Resilience module for fault tolerance and error handling.

This module provides:
- Retry logic with exponential backoff (Tenacity)
- Timeout management
- Graceful degradation

Based on ReadyTensor Week 2 resilience best practices.
"""

try:
    from .retry_policy import (
        RetryPolicy,
        retry_with_policy,
        llm_retry,
        vector_search_retry,
        api_call_retry
    )
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False
    RetryPolicy = None
    retry_with_policy = None
    llm_retry = None
    vector_search_retry = None
    api_call_retry = None

try:
    from .timeout_manager import (
        TimeoutManager,
        TimeoutConfig,
        with_timeout
    )
    TIMEOUT_AVAILABLE = True
except ImportError:
    TIMEOUT_AVAILABLE = False
    TimeoutManager = None
    TimeoutConfig = None
    with_timeout = None

__all__ = [
    'RetryPolicy',
    'retry_with_policy',
    'llm_retry',
    'vector_search_retry',
    'api_call_retry',
    'TimeoutManager',
    'TimeoutConfig',
    'with_timeout',
    'RETRY_AVAILABLE',
    'TIMEOUT_AVAILABLE',
]
