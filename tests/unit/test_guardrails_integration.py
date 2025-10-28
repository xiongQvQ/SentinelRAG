"""
Unit tests for Guardrails AI integration
Tests input/output validation, security features, and integration with audit logging
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.security.guardrails_integration import (
    GuardrailsValidator,
    GuardrailsConfig,
    create_default_validator,
    validate_user_input
)

# Skip all tests if guardrails-ai is not installed
pytest.importorskip("guardrails", reason="guardrails-ai not installed")


@pytest.fixture
def mock_audit_logger():
    """Create mock audit logger"""
    logger = Mock()
    logger.log_event = Mock()
    logger.log_security_violation = Mock()
    logger.EventType = Mock()
    logger.SeverityLevel = Mock()
    return logger


@pytest.fixture
def guardrails_config():
    """Create default Guardrails configuration"""
    return GuardrailsConfig(
        enable_input_validation=True,
        enable_output_validation=True,
        check_toxic_language=True,
        check_unusual_prompts=True,
        detect_pii=True,
        toxic_threshold=0.5,
        input_fail_action="exception",
        output_fail_action="filter"
    )


@pytest.fixture
def mock_guard():
    """Create mock Guardrails Guard"""
    guard = Mock()
    guard.use = Mock(return_value=guard)
    guard.validate = Mock()
    return guard


@pytest.mark.unit
@pytest.mark.skipif(not pytest.importorskip("guardrails", reason="guardrails-ai not installed"),
                   reason="guardrails-ai not available")
class TestGuardrailsConfig:
    """Test Guardrails configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = GuardrailsConfig()

        assert config.enable_input_validation is True
        assert config.enable_output_validation is True
        assert config.check_toxic_language is True
        assert config.check_unusual_prompts is True
        assert config.detect_pii is True
        assert config.toxic_threshold == 0.5
        assert config.input_fail_action == "exception"
        assert config.output_fail_action == "filter"

    def test_custom_config(self):
        """Test custom configuration"""
        config = GuardrailsConfig(
            enable_input_validation=False,
            toxic_threshold=0.7,
            input_fail_action="filter"
        )

        assert config.enable_input_validation is False
        assert config.toxic_threshold == 0.7
        assert config.input_fail_action == "filter"


@pytest.mark.unit
@patch('src.security.guardrails_integration.Guard')
@patch('src.security.guardrails_integration.ToxicLanguage')
@patch('src.security.guardrails_integration.UnusualPrompt')
@patch('src.security.guardrails_integration.DetectPII')
class TestGuardrailsValidator:
    """Test Guardrails validator"""

    def test_validator_initialization(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class, guardrails_config, mock_audit_logger):
        """Test validator initialization with all components"""
        # Setup mocks
        mock_guard = Mock()
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator
        validator = GuardrailsValidator(config=guardrails_config, audit_logger=mock_audit_logger)

        # Verify initialization
        assert validator.config == guardrails_config
        assert validator.audit_logger == mock_audit_logger
        assert validator.enable_guardrails is True

        # Verify stats initialization
        assert validator.total_validations == 0
        assert validator.failed_validations == 0
        assert validator.validation_by_type['input']['total'] == 0
        assert validator.validation_by_type['output']['total'] == 0

    def test_validate_input_success(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class, guardrails_config):
        """Test successful input validation"""
        # Setup mock guard to return successful validation
        mock_guard = Mock()
        mock_result = Mock()
        mock_result.validated_output = "clean text"
        mock_guard.validate = Mock(return_value=mock_result)
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator
        validator = GuardrailsValidator(config=guardrails_config)
        validator.input_guard = mock_guard

        # Validate input
        result = validator.validate_input("What is machine learning?")

        # Verify result
        assert result['valid'] is True
        assert result['validation_passed'] is True
        assert len(result['errors']) == 0
        assert validator.validation_by_type['input']['total'] == 1
        assert validator.validation_by_type['input']['failed'] == 0

    def test_validate_input_failure(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class, guardrails_config):
        """Test input validation failure"""
        # Setup mock guard to raise exception
        mock_guard = Mock()
        mock_guard.validate = Mock(side_effect=Exception("Toxic language detected"))
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator
        validator = GuardrailsValidator(config=guardrails_config)
        validator.input_guard = mock_guard

        # Validate input (should fail)
        result = validator.validate_input("This is toxic content")

        # Verify result
        assert result['valid'] is False
        assert result['validation_passed'] is False
        assert len(result['errors']) > 0
        assert validator.validation_by_type['input']['failed'] == 1

    def test_validate_output_success(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class, guardrails_config):
        """Test successful output validation"""
        # Setup mock guard
        mock_guard = Mock()
        mock_result = Mock()
        mock_result.validated_output = "Safe output"
        mock_guard.validate = Mock(return_value=mock_result)
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator
        validator = GuardrailsValidator(config=guardrails_config)
        validator.output_guard = mock_guard

        # Validate output
        result = validator.validate_output("Machine learning is a subset of AI")

        # Verify result
        assert result['valid'] is True
        assert result['validation_passed'] is True
        assert len(result['errors']) == 0
        assert validator.validation_by_type['output']['total'] == 1

    def test_validate_output_with_pii(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class, guardrails_config):
        """Test output validation with PII detection"""
        # Setup mock guard to detect PII
        mock_guard = Mock()
        mock_guard.validate = Mock(side_effect=Exception("PII detected: email"))
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator
        validator = GuardrailsValidator(config=guardrails_config)
        validator.output_guard = mock_guard

        # Validate output with PII
        result = validator.validate_output("Contact us at john@example.com")

        # Verify result
        assert result['valid'] is False
        assert result['validated_text'] == '[OUTPUT FILTERED]'
        assert len(result['errors']) > 0

    def test_validate_both(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class, guardrails_config):
        """Test validation of both input and output"""
        # Setup mocks
        mock_guard = Mock()
        mock_result = Mock()
        mock_result.validated_output = "validated text"
        mock_guard.validate = Mock(return_value=mock_result)
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator
        validator = GuardrailsValidator(config=guardrails_config)
        validator.input_guard = mock_guard
        validator.output_guard = mock_guard

        # Validate both
        result = validator.validate_both(
            user_input="What is ML?",
            llm_output="Machine learning is AI"
        )

        # Verify result
        assert 'input_validation' in result
        assert 'output_validation' in result
        assert result['both_valid'] is True
        assert 'validated_input' in result
        assert 'validated_output' in result

    def test_get_stats(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class, guardrails_config):
        """Test getting validation statistics"""
        # Setup
        mock_guard = Mock()
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator
        validator = GuardrailsValidator(config=guardrails_config)

        # Manually set some stats
        validator.total_validations = 10
        validator.failed_validations = 2
        validator.validation_by_type['input']['total'] = 6
        validator.validation_by_type['output']['total'] = 4

        # Get stats
        stats = validator.get_stats()

        # Verify stats
        assert stats['total_validations'] == 10
        assert stats['failed_validations'] == 2
        assert stats['success_rate'] == 0.8
        assert stats['input_validations']['total'] == 6
        assert stats['output_validations']['total'] == 4
        assert 'config' in stats

    def test_reset_stats(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class, guardrails_config):
        """Test resetting statistics"""
        # Setup
        mock_guard = Mock()
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        validator = GuardrailsValidator(config=guardrails_config)

        # Set some stats
        validator.total_validations = 10
        validator.failed_validations = 2

        # Reset
        validator.reset_stats()

        # Verify reset
        assert validator.total_validations == 0
        assert validator.failed_validations == 0

    def test_disabled_input_validation(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class):
        """Test with input validation disabled"""
        config = GuardrailsConfig(enable_input_validation=False)

        mock_guard = Mock()
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        validator = GuardrailsValidator(config=config)

        # Input guard should be None
        assert validator.input_guard is None

        # Validation should pass through
        result = validator.validate_input("any text")
        assert result['valid'] is True
        assert result['metadata']['guard_disabled'] is True

    def test_disabled_output_validation(self, mock_pii, mock_unusual, mock_toxic, mock_guard_class):
        """Test with output validation disabled"""
        config = GuardrailsConfig(enable_output_validation=False)

        mock_guard = Mock()
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        validator = GuardrailsValidator(config=config)

        # Output guard should be None
        assert validator.output_guard is None

        # Validation should pass through
        result = validator.validate_output("any output")
        assert result['valid'] is True
        assert result['metadata']['guard_disabled'] is True


@pytest.mark.unit
@patch('src.security.guardrails_integration.GuardrailsValidator')
class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_default_validator(self, mock_validator_class):
        """Test creating default validator"""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator

        validator = create_default_validator()

        assert validator == mock_validator
        mock_validator_class.assert_called_once()

    def test_validate_user_input_success(self, mock_validator_class):
        """Test validate_user_input with valid input"""
        mock_validator = Mock()
        mock_validator.validate_input = Mock(return_value={
            'valid': True,
            'validated_text': 'clean input',
            'errors': []
        })
        mock_validator_class.return_value = mock_validator

        result = validate_user_input("What is AI?")

        assert result == 'clean input'

    def test_validate_user_input_failure(self, mock_validator_class):
        """Test validate_user_input with invalid input"""
        mock_validator = Mock()
        mock_validator.validate_input = Mock(return_value={
            'valid': False,
            'validated_text': '',
            'errors': ['Toxic language detected']
        })
        mock_validator_class.return_value = mock_validator

        with pytest.raises(ValueError, match="Input validation failed"):
            validate_user_input("toxic input")


@pytest.mark.unit
class TestGuardrailsIntegrationWithAuditLogger:
    """Test Guardrails integration with audit logger"""

    @patch('src.security.guardrails_integration.Guard')
    @patch('src.security.guardrails_integration.ToxicLanguage')
    @patch('src.security.guardrails_integration.UnusualPrompt')
    def test_audit_logging_on_success(self, mock_unusual, mock_toxic, mock_guard_class, guardrails_config, mock_audit_logger):
        """Test that successful validations are logged"""
        # Setup
        mock_guard = Mock()
        mock_result = Mock()
        mock_result.validated_output = "clean"
        mock_guard.validate = Mock(return_value=mock_result)
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator with audit logger
        validator = GuardrailsValidator(config=guardrails_config, audit_logger=mock_audit_logger)
        validator.input_guard = mock_guard

        # Validate
        validator.validate_input("test input")

        # Verify audit logger was called
        assert mock_audit_logger.log_event.called

    @patch('src.security.guardrails_integration.Guard')
    def test_audit_logging_on_failure(self, mock_guard_class, guardrails_config, mock_audit_logger):
        """Test that validation failures are logged as security violations"""
        # Setup
        mock_guard = Mock()
        mock_guard.validate = Mock(side_effect=Exception("Validation failed"))
        mock_guard.use = Mock(return_value=mock_guard)
        mock_guard_class.return_value = mock_guard

        # Create validator with audit logger
        validator = GuardrailsValidator(config=guardrails_config, audit_logger=mock_audit_logger)
        validator.input_guard = mock_guard

        # Validate (should fail)
        validator.validate_input("bad input")

        # Verify security violation was logged
        assert mock_audit_logger.log_security_violation.called
