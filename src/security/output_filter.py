"""
Output Filter for RAG System
Filters and sanitizes outputs to prevent information leakage and harmful content
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for output filtering"""
    filter_pii: bool = True
    filter_credentials: bool = True
    filter_profanity: bool = False
    max_output_length: int = 5000
    redact_emails: bool = True
    redact_phones: bool = True
    redact_ssn: bool = True
    redact_credit_cards: bool = True
    custom_patterns: List[str] = None

    def __post_init__(self):
        if self.custom_patterns is None:
            self.custom_patterns = []


class OutputFilter:
    """
    Filters outputs for security and compliance

    Features:
    - PII detection and redaction
    - Credential detection
    - Profanity filtering
    - Length limiting
    - Custom pattern filtering
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize output filter

        Args:
            config: Filter configuration (uses defaults if None)
        """
        self.config = config or FilterConfig()
        self.redaction_count = 0

        # Compile PII patterns
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        self._phone_pattern = re.compile(
            r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
        )

        self._ssn_pattern = re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
        )

        self._credit_card_pattern = re.compile(
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        )

        # API key patterns
        self._api_key_patterns = [
            re.compile(r'api[_-]?key[\s]*[:=][\s]*[\'"]?([a-zA-Z0-9_\-]{20,})[\'"]?', re.IGNORECASE),
            re.compile(r'sk-[a-zA-Z0-9]{20,}'),  # OpenAI style
            re.compile(r'AIza[a-zA-Z0-9_\-]{35}'),  # Google API
            re.compile(r'AKIA[0-9A-Z]{16}'),  # AWS Access Key
        ]

        # Password patterns
        self._password_patterns = [
            re.compile(r'password[\s]*[:=][\s]*[\'"]?([^\s\'"]+)[\'"]?', re.IGNORECASE),
            re.compile(r'passwd[\s]*[:=][\s]*[\'"]?([^\s\'"]+)[\'"]?', re.IGNORECASE),
            re.compile(r'pwd[\s]*[:=][\s]*[\'"]?([^\s\'"]+)[\'"]?', re.IGNORECASE),
        ]

        # Basic profanity list (can be extended)
        self._profanity_words = {
            'damn', 'hell', 'crap', 'shit', 'fuck', 'ass', 'bitch'
        }

        # Compile custom patterns
        self._custom_compiled = [
            re.compile(pattern) for pattern in self.config.custom_patterns
        ]

    def filter(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Filter output text

        Args:
            text: Output text to filter
            context: Optional context for logging

        Returns:
            Dict with filtering results: {
                'filtered_text': str,
                'redactions': List[str],
                'warnings': List[str],
                'is_safe': bool
            }
        """
        context = context or "output"
        redactions = []
        warnings = []
        filtered_text = text

        if text is None:
            return {
                'filtered_text': '',
                'redactions': [],
                'warnings': ['Text is None'],
                'is_safe': True
            }

        # PII filtering
        if self.config.filter_pii:
            # Email redaction
            if self.config.redact_emails:
                if self._email_pattern.search(filtered_text):
                    filtered_text = self._email_pattern.sub('[EMAIL_REDACTED]', filtered_text)
                    redactions.append('email')

            # Phone number redaction
            if self.config.redact_phones:
                if self._phone_pattern.search(filtered_text):
                    filtered_text = self._phone_pattern.sub('[PHONE_REDACTED]', filtered_text)
                    redactions.append('phone')

            # SSN redaction
            if self.config.redact_ssn:
                if self._ssn_pattern.search(filtered_text):
                    filtered_text = self._ssn_pattern.sub('[SSN_REDACTED]', filtered_text)
                    redactions.append('ssn')

            # Credit card redaction
            if self.config.redact_credit_cards:
                if self._credit_card_pattern.search(filtered_text):
                    filtered_text = self._credit_card_pattern.sub('[CC_REDACTED]', filtered_text)
                    redactions.append('credit_card')

        # Credential filtering
        if self.config.filter_credentials:
            # API keys
            for pattern in self._api_key_patterns:
                if pattern.search(filtered_text):
                    filtered_text = pattern.sub('[API_KEY_REDACTED]', filtered_text)
                    redactions.append('api_key')
                    logger.warning(f"API key detected in {context}")

            # Passwords
            for pattern in self._password_patterns:
                if pattern.search(filtered_text):
                    filtered_text = pattern.sub('password=[PASSWORD_REDACTED]', filtered_text)
                    redactions.append('password')
                    logger.warning(f"Password detected in {context}")

        # Profanity filtering
        if self.config.filter_profanity:
            for word in self._profanity_words:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                if pattern.search(filtered_text):
                    filtered_text = pattern.sub('[REDACTED]', filtered_text)
                    redactions.append('profanity')

        # Custom patterns
        for pattern in self._custom_compiled:
            if pattern.search(filtered_text):
                filtered_text = pattern.sub('[REDACTED]', filtered_text)
                redactions.append('custom_pattern')

        # Length limiting
        if len(filtered_text) > self.config.max_output_length:
            filtered_text = filtered_text[:self.config.max_output_length] + '... [TRUNCATED]'
            warnings.append(f'Output truncated to {self.config.max_output_length} characters')

        # Update redaction count
        self.redaction_count += len(redactions)

        # Determine if output is safe
        is_safe = 'password' not in redactions and 'api_key' not in redactions

        result = {
            'filtered_text': filtered_text,
            'redactions': redactions,
            'warnings': warnings,
            'is_safe': is_safe,
            'original_length': len(text),
            'filtered_length': len(filtered_text)
        }

        # Log if redactions occurred
        if redactions:
            logger.info(f"Output filtering for {context}: {len(redactions)} redactions ({redactions})")

        return result

    def filter_strict(self, text: str, context: Optional[str] = None) -> str:
        """
        Strictly filter output and return filtered text

        Args:
            text: Output text to filter
            context: Optional context for logging

        Returns:
            Filtered text
        """
        result = self.filter(text, context)
        return result['filtered_text']

    def check_safe(self, text: str) -> bool:
        """
        Check if text is safe to output (no sensitive data)

        Args:
            text: Text to check

        Returns:
            True if safe, False otherwise
        """
        result = self.filter(text)
        return result['is_safe']

    def add_custom_pattern(self, pattern: str):
        """
        Add a custom pattern to filter

        Args:
            pattern: Regex pattern string
        """
        self.config.custom_patterns.append(pattern)
        self._custom_compiled.append(re.compile(pattern))
        logger.info(f"Added custom filter pattern: {pattern}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get filter statistics

        Returns:
            Dictionary with filter stats
        """
        return {
            'total_redactions': self.redaction_count,
            'filter_pii': self.config.filter_pii,
            'filter_credentials': self.config.filter_credentials,
            'filter_profanity': self.config.filter_profanity,
            'max_output_length': self.config.max_output_length,
            'custom_patterns_count': len(self._custom_compiled)
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.redaction_count = 0


# Convenience function
def filter_output(text: str) -> str:
    """
    Convenience function to filter output

    Args:
        text: Output text to filter

    Returns:
        Filtered text
    """
    filter_instance = OutputFilter()
    return filter_instance.filter_strict(text)


if __name__ == "__main__":
    # Example usage
    output_filter = OutputFilter()

    # Test PII filtering
    text_with_pii = "Contact me at john.doe@example.com or call 555-123-4567"
    result = output_filter.filter(text_with_pii)
    print("PII filtering:", result)

    # Test credential filtering
    text_with_creds = "My API key is sk-1234567890abcdef and password=secret123"
    result = output_filter.filter(text_with_creds)
    print("Credential filtering:", result)

    # Test safe check
    safe_text = "Machine learning is a subset of artificial intelligence"
    print("Is safe:", output_filter.check_safe(safe_text))
