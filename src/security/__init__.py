"""
Security module for RAG System
Provides input validation, output filtering, rate limiting, audit logging, and Guardrails AI integration
"""

from .input_validator import InputValidator, ValidationError
from .output_filter import OutputFilter
from .rate_limiter import RateLimiter, RateLimitExceeded
from .audit_logger import AuditLogger

try:
    from .guardrails_integration import (
        GuardrailsValidator,
        GuardrailsConfig,
        create_default_validator,
        validate_user_input
    )
    HAS_GUARDRAILS = True
except ImportError:
    HAS_GUARDRAILS = False

__all__ = [
    'InputValidator',
    'ValidationError',
    'OutputFilter',
    'RateLimiter',
    'RateLimitExceeded',
    'AuditLogger',
]

if HAS_GUARDRAILS:
    __all__.extend([
        'GuardrailsValidator',
        'GuardrailsConfig',
        'create_default_validator',
        'validate_user_input'
    ])
