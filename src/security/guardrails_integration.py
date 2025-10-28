"""
Guardrails AI Integration for RAG System
Implements runtime safety and output validation using Guardrails AI framework

Based on ReadyTensor Week 9 Lesson 5:
https://app.readytensor.ai/lessons/guardrails-in-action-runtime-safety-and-output-validation-for-agentic-ai-aaidc-week9-lesson5-tiBt9Nevyqrw
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

try:
    from guardrails import Guard, OnFailAction
    from guardrails.hub import (
        ToxicLanguage,
        UnusualPrompt,
        DetectPII,
    )
    HAS_GUARDRAILS = True
except ImportError:
    HAS_GUARDRAILS = False
    Guard = None  # Placeholder for type hints
    OnFailAction = None
    logging.warning("guardrails-ai not installed. Please install: pip install guardrails-ai")

# Import audit logger enums
try:
    from .audit_logger import EventType, SeverityLevel
except ImportError:
    try:
        from audit_logger import EventType, SeverityLevel
    except ImportError:
        # Fallback if audit_logger not available
        EventType = None
        SeverityLevel = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuardrailType(Enum):
    """Types of guardrails"""
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


class FailureAction(Enum):
    """Actions to take on guardrail failure"""
    EXCEPTION = "exception"  # Raise an exception
    FILTER = "filter"  # Filter out the problematic content
    REFRAIN = "refrain"  # Refuse to process
    FIX = "fix"  # Attempt to fix the content
    REASK = "reask"  # Ask user to rephrase


@dataclass
class GuardrailsConfig:
    """Configuration for Guardrails AI"""
    # Input validation settings
    enable_input_validation: bool = True
    check_toxic_language: bool = True
    check_unusual_prompts: bool = True
    toxic_threshold: float = 0.5

    # Output validation settings
    enable_output_validation: bool = True
    detect_pii: bool = True
    check_hallucination: bool = False  # Requires additional setup

    # Failure handling
    input_fail_action: str = "exception"
    output_fail_action: str = "filter"

    # LLM Configuration (for validation)
    llm_model: str = "gemini/gemini-2.5-flash"  # Use Gemini via LiteLLM
    llm_api_key: Optional[str] = None  # Google API key
    llm_api_base: Optional[str] = None  # Optional API base URL

    # Logging
    log_validations: bool = True
    log_failures: bool = True


class GuardrailsValidator:
    """
    Guardrails AI validator for RAG system

    Features:
    - Input validation (toxic language, unusual prompts)
    - Output validation (PII detection, hallucination check)
    - Configurable failure actions
    - Integration with audit logging

    Usage:
        validator = GuardrailsValidator()

        # Validate user input
        validated_input = validator.validate_input(user_query)

        # Validate LLM output
        validated_output = validator.validate_output(llm_response, context=user_query)
    """

    def __init__(self, config: Optional[GuardrailsConfig] = None, audit_logger=None, enabled: bool = True):
        """
        Initialize Guardrails validator

        Args:
            config: Guardrails configuration
            audit_logger: Optional audit logger instance
            enabled: Whether to enable Guardrails validation (default: True)
        """
        self.enabled = enabled and HAS_GUARDRAILS

        if enabled and not HAS_GUARDRAILS:
            logger.warning(
                "guardrails-ai is not installed. Guardrails validation disabled. "
                "Install with: pip install guardrails-ai"
            )

        self.config = config or GuardrailsConfig()
        self.audit_logger = audit_logger

        # Initialize guards only if enabled
        if self.enabled:
            self.input_guard = self._create_input_guard()
            self.output_guard = self._create_output_guard()
        else:
            self.input_guard = None
            self.output_guard = None

        # Statistics
        self.total_validations = 0
        self.failed_validations = 0
        self.validation_by_type = {
            'input': {'total': 0, 'failed': 0},
            'output': {'total': 0, 'failed': 0}
        }

    def _create_input_guard(self) -> Optional[Guard]:
        """Create input validation guard"""
        if not self.config.enable_input_validation:
            return None

        try:
            import os

            # Configure environment for LiteLLM to use Gemini
            if self.config.llm_api_key:
                os.environ['GOOGLE_API_KEY'] = self.config.llm_api_key

            # Set LiteLLM to use Gemini (via environment variable)
            # LiteLLM will use gemini/ prefix to route to Google's Gemini API
            if self.config.llm_model:
                os.environ['LITELLM_MODEL'] = self.config.llm_model

            # Create guard (without model parameter)
            guard = Guard()

            # Add toxic language detector
            if self.config.check_toxic_language:
                guard = guard.use(
                    ToxicLanguage(
                        threshold=self.config.toxic_threshold,
                        on_fail=self._get_fail_action(self.config.input_fail_action)
                    )
                )
                logger.info(f"Added ToxicLanguage validator to input guard (will use {self.config.llm_model})")

            # Add unusual prompt detector
            if self.config.check_unusual_prompts:
                guard = guard.use(
                    UnusualPrompt(
                        llm_callable=self.config.llm_model,  # 指定使用 Gemini
                        on_fail=self._get_fail_action(self.config.input_fail_action)
                    )
                )
                logger.info(f"Added UnusualPrompt validator to input guard (using {self.config.llm_model})")

            return guard

        except Exception as e:
            logger.error(f"Failed to create input guard: {e}")
            logger.info("Make sure validators are installed: guardrails hub install hub://guardrails/toxic_language hub://guardrails/unusual_prompt")
            return None

    def _create_output_guard(self) -> Optional[Guard]:
        """Create output validation guard"""
        if not self.config.enable_output_validation:
            return None

        try:
            import os

            # Configure environment for LiteLLM to use Gemini
            if self.config.llm_api_key:
                os.environ['GOOGLE_API_KEY'] = self.config.llm_api_key

            # Set LiteLLM to use Gemini (via environment variable)
            if self.config.llm_model:
                os.environ['LITELLM_MODEL'] = self.config.llm_model

            # Create guard (without model parameter)
            guard = Guard()

            # Add PII detector
            if self.config.detect_pii:
                guard = guard.use(
                    DetectPII(
                        on_fail=self._get_fail_action(self.config.output_fail_action)
                    )
                )
                logger.info(f"Added DetectPII validator to output guard (will use {self.config.llm_model})")

            return guard

        except Exception as e:
            logger.error(f"Failed to create output guard: {e}")
            logger.info("Make sure validators are installed: guardrails hub install hub://guardrails/detect_pii")
            return None

    def _get_fail_action(self, action_str: str):
        """Convert failure action string to OnFailAction"""
        action_map = {
            'exception': OnFailAction.EXCEPTION,
            'filter': OnFailAction.FILTER,
            'refrain': OnFailAction.REFRAIN,
            'fix': OnFailAction.FIX,
            'reask': OnFailAction.REASK,
        }
        return action_map.get(action_str.lower(), OnFailAction.EXCEPTION)

    def validate_input(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate user input using Guardrails

        Args:
            text: Input text to validate
            context: Optional context for logging

        Returns:
            Dict with validation results: {
                'valid': bool,
                'validated_text': str,
                'validation_passed': bool,
                'errors': List[str],
                'metadata': Dict
            }
        """
        context = context or "user_input"
        self.validation_by_type['input']['total'] += 1
        self.total_validations += 1

        if self.input_guard is None:
            # No guard configured, pass through
            return {
                'valid': True,
                'validated_text': text,
                'validation_passed': True,
                'errors': [],
                'metadata': {'guard_disabled': True}
            }

        try:
            # Run validation
            result = self.input_guard.validate(text)

            # Log success
            if self.config.log_validations:
                logger.info(f"Input validation passed for {context}")

            if self.audit_logger and EventType and SeverityLevel:
                self.audit_logger.log_event(
                    event_type=EventType.ACCESS,
                    severity=SeverityLevel.INFO,
                    action="input_validation_success",
                    message=f"Input validation passed for {context}",
                    status="success"
                )

            return {
                'valid': True,
                'validated_text': result.validated_output if hasattr(result, 'validated_output') else text,
                'validation_passed': True,
                'errors': [],
                'metadata': {
                    'guard_type': 'input',
                    'validators_used': ['ToxicLanguage', 'UnusualPrompt']
                }
            }

        except Exception as e:
            # Validation failed
            self.validation_by_type['input']['failed'] += 1
            self.failed_validations += 1

            error_msg = str(e)

            if self.config.log_failures:
                logger.warning(f"Input validation failed for {context}: {error_msg}")

            if self.audit_logger:
                self.audit_logger.log_security_violation(
                    violation_type="input_validation_failure",
                    details=error_msg,
                    metadata={'context': context}
                )

            return {
                'valid': False,
                'validated_text': '',
                'validation_passed': False,
                'errors': [error_msg],
                'metadata': {
                    'guard_type': 'input',
                    'failure_action': self.config.input_fail_action
                }
            }

    def validate_output(self, text: str, context: Optional[str] = None,
                       user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate LLM output using Guardrails

        Args:
            text: Output text to validate
            context: Optional context for logging
            user_query: Original user query for hallucination detection

        Returns:
            Dict with validation results
        """
        context = context or "llm_output"
        self.validation_by_type['output']['total'] += 1
        self.total_validations += 1

        if self.output_guard is None:
            # No guard configured, pass through
            return {
                'valid': True,
                'validated_text': text,
                'validation_passed': True,
                'errors': [],
                'metadata': {'guard_disabled': True}
            }

        try:
            # Run validation
            result = self.output_guard.validate(text)

            # Log success
            if self.config.log_validations:
                logger.info(f"Output validation passed for {context}")

            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type=self.audit_logger.EventType.RESPONSE,
                    severity=self.audit_logger.SeverityLevel.INFO,
                    action="output_validation_success",
                    message=f"Output validation passed for {context}",
                    status="success"
                )

            return {
                'valid': True,
                'validated_text': result.validated_output if hasattr(result, 'validated_output') else text,
                'validation_passed': True,
                'errors': [],
                'metadata': {
                    'guard_type': 'output',
                    'validators_used': ['DetectPII']
                }
            }

        except Exception as e:
            # Validation failed
            self.validation_by_type['output']['failed'] += 1
            self.failed_validations += 1

            error_msg = str(e)

            if self.config.log_failures:
                logger.warning(f"Output validation failed for {context}: {error_msg}")

            if self.audit_logger:
                self.audit_logger.log_security_violation(
                    violation_type="output_validation_failure",
                    details=error_msg,
                    metadata={'context': context, 'user_query': user_query}
                )

            return {
                'valid': False,
                'validated_text': '[OUTPUT FILTERED]',
                'validation_passed': False,
                'errors': [error_msg],
                'metadata': {
                    'guard_type': 'output',
                    'failure_action': self.config.output_fail_action
                }
            }

    def validate_both(self, user_input: str, llm_output: str) -> Dict[str, Any]:
        """
        Validate both input and output

        Args:
            user_input: User query
            llm_output: LLM response

        Returns:
            Combined validation results
        """
        input_result = self.validate_input(user_input, context="combined_validation_input")
        output_result = self.validate_output(llm_output, context="combined_validation_output", user_query=user_input)

        return {
            'input_validation': input_result,
            'output_validation': output_result,
            'both_valid': input_result['valid'] and output_result['valid'],
            'validated_input': input_result['validated_text'],
            'validated_output': output_result['validated_text']
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics

        Returns:
            Dictionary with validation stats
        """
        return {
            'total_validations': self.total_validations,
            'failed_validations': self.failed_validations,
            'success_rate': (self.total_validations - self.failed_validations) / max(1, self.total_validations),
            'input_validations': self.validation_by_type['input'],
            'output_validations': self.validation_by_type['output'],
            'config': {
                'input_validation_enabled': self.config.enable_input_validation,
                'output_validation_enabled': self.config.enable_output_validation,
                'toxic_threshold': self.config.toxic_threshold,
                'input_fail_action': self.config.input_fail_action,
                'output_fail_action': self.config.output_fail_action
            }
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.total_validations = 0
        self.failed_validations = 0
        self.validation_by_type = {
            'input': {'total': 0, 'failed': 0},
            'output': {'total': 0, 'failed': 0}
        }


# Convenience functions
def create_default_validator(audit_logger=None) -> GuardrailsValidator:
    """
    Create validator with default configuration

    Args:
        audit_logger: Optional audit logger instance

    Returns:
        Configured GuardrailsValidator
    """
    return GuardrailsValidator(audit_logger=audit_logger)


def validate_user_input(text: str, validator: Optional[GuardrailsValidator] = None) -> str:
    """
    Convenience function to validate user input

    Args:
        text: Input text
        validator: Optional validator instance

    Returns:
        Validated text

    Raises:
        Exception: If validation fails with exception action
    """
    if validator is None:
        validator = create_default_validator()

    result = validator.validate_input(text)

    if not result['valid']:
        raise ValueError(f"Input validation failed: {'; '.join(result['errors'])}")

    return result['validated_text']


if __name__ == "__main__":
    # Example usage
    print("Guardrails AI Integration Example")
    print("=" * 50)

    if not HAS_GUARDRAILS:
        print("\n⚠️  guardrails-ai not installed!")
        print("Install with: pip install guardrails-ai")
        print("Then install validators:")
        print("  guardrails hub install hub://guardrails/toxic_language")
        print("  guardrails hub install hub://guardrails/unusual_prompt")
        print("  guardrails hub install hub://guardrails/detect_pii")
    else:
        try:
            # Create validator
            validator = create_default_validator()

            # Test input validation
            print("\n✅ Testing Input Validation:")
            safe_input = "What is machine learning?"
            result = validator.validate_input(safe_input)
            print(f"  Input: {safe_input}")
            print(f"  Valid: {result['valid']}")

            # Test output validation
            print("\n✅ Testing Output Validation:")
            safe_output = "Machine learning is a subset of artificial intelligence."
            result = validator.validate_output(safe_output)
            print(f"  Output: {safe_output}")
            print(f"  Valid: {result['valid']}")

            # Print statistics
            print("\n📊 Statistics:")
            import json
            print(json.dumps(validator.get_stats(), indent=2))

        except Exception as e:
            print(f"\n⚠️  Error during example: {e}")
            print("\nMake sure to install validators first:")
            print("  guardrails hub install hub://guardrails/toxic_language")
            print("  guardrails hub install hub://guardrails/unusual_prompt")
            print("  guardrails hub install hub://guardrails/detect_pii")
