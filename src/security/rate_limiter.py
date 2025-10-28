"""
Rate Limiter for RAG System
Implements rate limiting to prevent abuse and ensure fair usage
"""

import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 10
    requests_per_hour: int = 100
    requests_per_day: int = 1000
    enable_user_limits: bool = True
    enable_ip_limits: bool = True
    burst_size: int = 5  # Allow burst requests


class RateLimiter:
    """
    Rate limiter using token bucket algorithm

    Features:
    - Per-user rate limiting
    - Per-IP rate limiting
    - Multiple time windows (minute, hour, day)
    - Burst handling
    - Thread-safe
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter

        Args:
            config: Rate limit configuration (uses defaults if None)
        """
        self.config = config or RateLimitConfig()

        # Track requests per user/IP
        self._user_requests = defaultdict(lambda: {
            'minute': deque(),
            'hour': deque(),
            'day': deque()
        })

        self._ip_requests = defaultdict(lambda: {
            'minute': deque(),
            'hour': deque(),
            'day': deque()
        })

        # Thread safety
        self._lock = Lock()

        # Statistics
        self.total_requests = 0
        self.blocked_requests = 0

    def check_rate_limit(self, identifier: str, identifier_type: str = 'user') -> bool:
        """
        Check if request is within rate limits

        Args:
            identifier: User ID, IP address, or other identifier
            identifier_type: Type of identifier ('user' or 'ip')

        Returns:
            True if within limits, False if exceeded

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        with self._lock:
            current_time = time.time()

            # Choose the appropriate request tracker
            if identifier_type == 'user':
                if not self.config.enable_user_limits:
                    return True
                requests = self._user_requests[identifier]
            elif identifier_type == 'ip':
                if not self.config.enable_ip_limits:
                    return True
                requests = self._ip_requests[identifier]
            else:
                raise ValueError(f"Unknown identifier type: {identifier_type}")

            # Clean old requests and check limits
            windows = {
                'minute': (60, self.config.requests_per_minute),
                'hour': (3600, self.config.requests_per_hour),
                'day': (86400, self.config.requests_per_day)
            }

            for window_name, (window_seconds, limit) in windows.items():
                window_start = current_time - window_seconds
                request_queue = requests[window_name]

                # Remove old requests
                while request_queue and request_queue[0] < window_start:
                    request_queue.popleft()

                # Check limit
                if len(request_queue) >= limit:
                    self.blocked_requests += 1
                    logger.warning(
                        f"Rate limit exceeded for {identifier_type} {identifier} "
                        f"in {window_name} window: {len(request_queue)}/{limit}"
                    )
                    raise RateLimitExceeded(
                        f"Rate limit exceeded: {len(request_queue)}/{limit} "
                        f"requests in {window_name}"
                    )

            # Add current request to all windows
            for window_name in windows.keys():
                requests[window_name].append(current_time)

            self.total_requests += 1
            return True

    def try_acquire(self, identifier: str, identifier_type: str = 'user') -> bool:
        """
        Try to acquire permission for a request (non-raising version)

        Args:
            identifier: User ID, IP address, or other identifier
            identifier_type: Type of identifier ('user' or 'ip')

        Returns:
            True if request allowed, False if rate limit exceeded
        """
        try:
            return self.check_rate_limit(identifier, identifier_type)
        except RateLimitExceeded:
            return False

    def get_remaining(self, identifier: str, identifier_type: str = 'user',
                     window: str = 'minute') -> int:
        """
        Get remaining requests for an identifier in a time window

        Args:
            identifier: User ID, IP address, or other identifier
            identifier_type: Type of identifier ('user' or 'ip')
            window: Time window ('minute', 'hour', 'day')

        Returns:
            Number of remaining requests
        """
        with self._lock:
            current_time = time.time()

            # Choose the appropriate request tracker
            if identifier_type == 'user':
                requests = self._user_requests[identifier]
            elif identifier_type == 'ip':
                requests = self._ip_requests[identifier]
            else:
                return 0

            # Determine window parameters
            windows = {
                'minute': (60, self.config.requests_per_minute),
                'hour': (3600, self.config.requests_per_hour),
                'day': (86400, self.config.requests_per_day)
            }

            if window not in windows:
                return 0

            window_seconds, limit = windows[window]
            window_start = current_time - window_seconds
            request_queue = requests[window]

            # Remove old requests
            while request_queue and request_queue[0] < window_start:
                request_queue.popleft()

            # Calculate remaining
            remaining = max(0, limit - len(request_queue))
            return remaining

    def reset(self, identifier: Optional[str] = None, identifier_type: str = 'user'):
        """
        Reset rate limit counters

        Args:
            identifier: Specific identifier to reset (None to reset all)
            identifier_type: Type of identifier ('user' or 'ip')
        """
        with self._lock:
            if identifier is None:
                # Reset all
                if identifier_type == 'user':
                    self._user_requests.clear()
                elif identifier_type == 'ip':
                    self._ip_requests.clear()
                logger.info(f"Reset all {identifier_type} rate limits")
            else:
                # Reset specific identifier
                if identifier_type == 'user':
                    if identifier in self._user_requests:
                        del self._user_requests[identifier]
                elif identifier_type == 'ip':
                    if identifier in self._ip_requests:
                        del self._ip_requests[identifier]
                logger.info(f"Reset rate limit for {identifier_type}: {identifier}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics

        Returns:
            Dictionary with rate limiter stats
        """
        with self._lock:
            return {
                'total_requests': self.total_requests,
                'blocked_requests': self.blocked_requests,
                'block_rate': self.blocked_requests / max(1, self.total_requests),
                'tracked_users': len(self._user_requests),
                'tracked_ips': len(self._ip_requests),
                'config': {
                    'requests_per_minute': self.config.requests_per_minute,
                    'requests_per_hour': self.config.requests_per_hour,
                    'requests_per_day': self.config.requests_per_day,
                    'burst_size': self.config.burst_size
                }
            }

    def reset_stats(self):
        """Reset statistics counters"""
        with self._lock:
            self.total_requests = 0
            self.blocked_requests = 0


# Decorator for rate limiting
def rate_limit(identifier_func, identifier_type='user', limiter=None):
    """
    Decorator to add rate limiting to a function

    Args:
        identifier_func: Function to extract identifier from function arguments
        identifier_type: Type of identifier ('user' or 'ip')
        limiter: RateLimiter instance (creates new one if None)

    Returns:
        Decorated function

    Example:
        @rate_limit(lambda args, kwargs: kwargs.get('user_id', 'anonymous'))
        def my_function(user_id=None):
            pass
    """
    if limiter is None:
        limiter = RateLimiter()

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract identifier
            identifier = identifier_func(args, kwargs)

            # Check rate limit
            limiter.check_rate_limit(identifier, identifier_type)

            # Execute function
            return func(*args, **kwargs)

        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    limiter = RateLimiter(RateLimitConfig(
        requests_per_minute=5,
        requests_per_hour=20,
        requests_per_day=100
    ))

    # Test rate limiting
    user_id = "user123"

    try:
        for i in range(10):
            if limiter.try_acquire(user_id):
                print(f"Request {i+1} allowed")
                remaining = limiter.get_remaining(user_id)
                print(f"  Remaining: {remaining}")
            else:
                print(f"Request {i+1} blocked - rate limit exceeded")

            # Small delay
            time.sleep(0.1)
    except RateLimitExceeded as e:
        print(f"Rate limit error: {e}")

    # Print statistics
    print("\nStatistics:", limiter.get_stats())
