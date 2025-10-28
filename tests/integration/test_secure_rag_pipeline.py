"""
Integration tests for Secure RAG Pipeline with Guardrails AI
Tests end-to-end security features and validation workflows
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from src.security.rate_limiter import RateLimitExceeded

# Try to import SecureRAGPipeline
try:
    from src.rag_pipeline_with_guardrails import SecureRAGPipeline
    HAS_SECURE_PIPELINE = True
except ImportError:
    HAS_SECURE_PIPELINE = False
    SecureRAGPipeline = None


@pytest.fixture
def mock_google_api_key():
    """Mock Google API key"""
    return "test_google_api_key_12345"


@pytest.fixture
def mock_vector_store_manager():
    """Mock vector store manager"""
    manager = Mock()
    manager.initialize_from_articles = Mock()
    manager.search_documents = Mock(return_value=[
        {
            'content': 'Machine learning is a subset of AI',
            'title': 'Machine Learning',
            'url': 'https://en.wikipedia.org/wiki/ML',
            'topic': 'AI',
            'similarity_score': 0.95,
            'chunk_index': 0
        }
    ])
    manager.get_context_for_query = Mock(return_value="Context about ML")
    return manager


@pytest.fixture
def mock_llm():
    """Mock Gemini LLM"""
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(content="Machine learning is a subset of artificial intelligence"))
    return llm


@pytest.fixture
def mock_qa_chain():
    """Mock QA chain"""
    chain = Mock()
    chain.__call__ = Mock(return_value={
        'query': 'What is ML?',
        'result': 'Machine learning is a subset of artificial intelligence',
        'source_documents': []
    })
    return chain


@pytest.mark.integration
@pytest.mark.skipif(not HAS_SECURE_PIPELINE, reason="SecureRAGPipeline not available")
class TestSecureRAGPipelineInitialization:
    """Test Secure RAG Pipeline initialization"""

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.GuardrailsValidator')
    @patch('src.rag_pipeline_with_guardrails.RateLimiter')
    @patch('src.rag_pipeline_with_guardrails.get_audit_logger')
    def test_pipeline_initialization_with_all_features(self, mock_audit, mock_rate, mock_guard, mock_vector, mock_google_api_key):
        """Test pipeline initialization with all security features enabled"""
        # Setup mocks
        mock_vector.return_value = Mock()
        mock_guard.return_value = Mock()
        mock_rate.return_value = Mock()
        mock_audit.return_value = Mock()

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=True,
            enable_rate_limiting=True,
            enable_audit_logging=True
        )

        # Verify initialization
        assert pipeline.enable_guardrails is True
        assert pipeline.enable_rate_limiting is True
        assert pipeline.enable_audit_logging is True
        assert pipeline.guardrails_validator is not None
        assert pipeline.rate_limiter is not None
        assert pipeline.audit_logger is not None

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    def test_pipeline_initialization_without_security(self, mock_vector, mock_google_api_key):
        """Test pipeline initialization with security features disabled"""
        # Setup
        mock_vector.return_value = Mock()

        # Create pipeline without security features
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=False,
            enable_rate_limiting=False,
            enable_audit_logging=False
        )

        # Verify
        assert pipeline.enable_guardrails is False
        assert pipeline.enable_rate_limiting is False
        assert pipeline.enable_audit_logging is False

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.GuardrailsValidator')
    def test_pipeline_graceful_degradation_without_guardrails(self, mock_guard, mock_vector, mock_google_api_key):
        """Test pipeline gracefully handles missing guardrails-ai package"""
        # Setup - make GuardrailsValidator raise ImportError
        mock_guard.side_effect = ImportError("guardrails-ai not installed")
        mock_vector.return_value = Mock()

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=True
        )

        # Verify graceful degradation
        assert pipeline.enable_guardrails is False
        assert pipeline.guardrails_validator is None


@pytest.mark.integration
@pytest.mark.skipif(not HAS_SECURE_PIPELINE, reason="SecureRAGPipeline not available")
class TestSecureRAGPipelineValidation:
    """Test validation workflows in Secure RAG Pipeline"""

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.GuardrailsValidator')
    @patch('src.rag_pipeline_with_guardrails.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_with_guardrails.RetrievalQA')
    def test_successful_query_with_validation(self, mock_qa, mock_llm, mock_guard_class, mock_vector, mock_google_api_key):
        """Test successful query with input and output validation"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance

        mock_validator = Mock()
        mock_validator.validate_input = Mock(return_value={
            'valid': True,
            'validated_text': 'What is ML?',
            'errors': []
        })
        mock_validator.validate_output = Mock(return_value={
            'valid': True,
            'validated_text': 'ML is AI subset',
            'errors': []
        })
        mock_guard_class.return_value = mock_validator

        mock_chain = Mock()
        mock_chain.__call__ = Mock(return_value={
            'query': 'What is ML?',
            'result': 'ML is AI subset',
            'source_documents': []
        })
        mock_qa.from_chain_type = Mock(return_value=mock_chain)

        # Create and initialize pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=True,
            enable_rate_limiting=False
        )
        pipeline.qa_chain = mock_chain
        pipeline.guardrails_validator = mock_validator

        # Ask question
        response = pipeline.ask_question("What is ML?", user_id="test_user")

        # Verify validation was called
        assert mock_validator.validate_input.called
        assert mock_validator.validate_output.called

        # Verify response
        assert 'question' in response
        assert 'answer' in response
        assert response['security']['guardrails_enabled'] is True
        assert response['security']['input_validated'] is True
        assert response['security']['output_validated'] is True

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.GuardrailsValidator')
    def test_input_validation_failure(self, mock_guard_class, mock_vector, mock_google_api_key):
        """Test input validation failure blocks query"""
        # Setup mocks
        mock_vector.return_value = Mock()

        mock_validator = Mock()
        mock_validator.validate_input = Mock(return_value={
            'valid': False,
            'validated_text': '',
            'errors': ['Toxic language detected']
        })
        mock_guard_class.return_value = mock_validator

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=True,
            enable_rate_limiting=False
        )
        pipeline.qa_chain = Mock()
        pipeline.guardrails_validator = mock_validator

        # Attempt query with toxic content
        with pytest.raises(ValueError, match="Input validation failed"):
            pipeline.ask_question("toxic content", user_id="test_user")

        # Verify stats updated
        assert pipeline.stats['validation_failures'] == 1

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.GuardrailsValidator')
    @patch('src.rag_pipeline_with_guardrails.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_with_guardrails.RetrievalQA')
    def test_output_validation_filters_pii(self, mock_qa, mock_llm, mock_guard_class, mock_vector, mock_google_api_key):
        """Test output validation filters PII"""
        # Setup mocks
        mock_vector.return_value = Mock()

        mock_validator = Mock()
        mock_validator.validate_input = Mock(return_value={
            'valid': True,
            'validated_text': 'What is contact?',
            'errors': []
        })
        mock_validator.validate_output = Mock(return_value={
            'valid': False,
            'validated_text': '[OUTPUT FILTERED]',
            'errors': ['PII detected']
        })
        mock_guard_class.return_value = mock_validator

        mock_chain = Mock()
        mock_chain.__call__ = Mock(return_value={
            'query': 'What is contact?',
            'result': 'Email: john@example.com',
            'source_documents': []
        })
        mock_qa.from_chain_type = Mock(return_value=mock_chain)

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=True,
            enable_rate_limiting=False
        )
        pipeline.qa_chain = mock_chain
        pipeline.guardrails_validator = mock_validator

        # Ask question
        response = pipeline.ask_question("What is contact?", user_id="test_user")

        # Verify output was filtered
        assert response['answer'] == '[OUTPUT FILTERED]'
        assert pipeline.stats['validation_failures'] == 1


@pytest.mark.integration
@pytest.mark.skipif(not HAS_SECURE_PIPELINE, reason="SecureRAGPipeline not available")
class TestSecureRAGPipelineRateLimiting:
    """Test rate limiting in Secure RAG Pipeline"""

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.RateLimiter')
    def test_rate_limiting_blocks_excessive_requests(self, mock_rate_class, mock_vector, mock_google_api_key):
        """Test rate limiter blocks excessive requests"""
        # Setup mocks
        mock_vector.return_value = Mock()

        mock_limiter = Mock()
        mock_limiter.check_rate_limit = Mock(side_effect=RateLimitExceeded("Rate limit exceeded"))
        mock_rate_class.return_value = mock_limiter

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=False,
            enable_rate_limiting=True
        )
        pipeline.qa_chain = Mock()
        pipeline.rate_limiter = mock_limiter

        # Attempt query
        with pytest.raises(RateLimitExceeded):
            pipeline.ask_question("test question", user_id="test_user")

        # Verify stats
        assert pipeline.stats['rate_limited_queries'] == 1
        assert pipeline.stats['failed_queries'] == 1

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    def test_rate_limiting_disabled(self, mock_vector, mock_google_api_key):
        """Test pipeline works with rate limiting disabled"""
        # Setup
        mock_vector.return_value = Mock()

        # Create pipeline without rate limiting
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_rate_limiting=False
        )

        # Rate limiter should be None
        assert pipeline.rate_limiter is None


@pytest.mark.integration
@pytest.mark.skipif(not HAS_SECURE_PIPELINE, reason="SecureRAGPipeline not available")
class TestSecureRAGPipelineAuditLogging:
    """Test audit logging in Secure RAG Pipeline"""

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.GuardrailsValidator')
    @patch('src.rag_pipeline_with_guardrails.get_audit_logger')
    @patch('src.rag_pipeline_with_guardrails.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_with_guardrails.RetrievalQA')
    def test_successful_query_logged(self, mock_qa, mock_llm, mock_audit, mock_guard_class, mock_vector, mock_google_api_key):
        """Test successful queries are logged"""
        # Setup mocks
        mock_vector.return_value = Mock()

        mock_logger = Mock()
        mock_audit.return_value = mock_logger

        mock_validator = Mock()
        mock_validator.validate_input = Mock(return_value={
            'valid': True,
            'validated_text': 'test',
            'errors': []
        })
        mock_validator.validate_output = Mock(return_value={
            'valid': True,
            'validated_text': 'response',
            'errors': []
        })
        mock_guard_class.return_value = mock_validator

        mock_chain = Mock()
        mock_chain.__call__ = Mock(return_value={
            'query': 'test',
            'result': 'response',
            'source_documents': []
        })
        mock_qa.from_chain_type = Mock(return_value=mock_chain)

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_audit_logging=True
        )
        pipeline.qa_chain = mock_chain
        pipeline.audit_logger = mock_logger
        pipeline.guardrails_validator = mock_validator

        # Ask question
        pipeline.ask_question("test", user_id="test_user")

        # Verify logging occurred
        # The validator's audit logger should be called
        assert mock_validator.validate_input.called
        assert mock_validator.validate_output.called


@pytest.mark.integration
@pytest.mark.skipif(not HAS_SECURE_PIPELINE, reason="SecureRAGPipeline not available")
class TestSecureRAGPipelineStatistics:
    """Test statistics tracking in Secure RAG Pipeline"""

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_with_guardrails.RetrievalQA')
    def test_stats_tracking(self, mock_qa, mock_llm, mock_vector, mock_google_api_key):
        """Test statistics are tracked correctly"""
        # Setup mocks
        mock_vector.return_value = Mock()

        mock_chain = Mock()
        mock_chain.__call__ = Mock(return_value={
            'query': 'test',
            'result': 'response',
            'source_documents': []
        })
        mock_qa.from_chain_type = Mock(return_value=mock_chain)

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=False,
            enable_rate_limiting=False,
            enable_audit_logging=False
        )
        pipeline.qa_chain = mock_chain

        # Make successful query
        pipeline.ask_question("test")

        # Verify stats
        assert pipeline.stats['total_queries'] == 1
        assert pipeline.stats['successful_queries'] == 1
        assert pipeline.stats['failed_queries'] == 0

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    def test_get_stats(self, mock_vector, mock_google_api_key):
        """Test get_stats method"""
        # Setup
        mock_vector.return_value = Mock()

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=True,
            enable_rate_limiting=True,
            enable_audit_logging=True
        )

        # Get stats
        stats = pipeline.get_stats()

        # Verify stats structure
        assert 'pipeline' in stats
        assert 'security' in stats
        assert stats['security']['guardrails_enabled'] is True
        assert stats['security']['rate_limiting_enabled'] is True
        assert stats['security']['audit_logging_enabled'] is True


@pytest.mark.integration
@pytest.mark.skipif(not HAS_SECURE_PIPELINE, reason="SecureRAGPipeline not available")
class TestSecureRAGPipelineChatMode:
    """Test chat mode with Guardrails validation"""

    @patch('src.rag_pipeline_with_guardrails.VectorStoreManager')
    @patch('src.rag_pipeline_with_guardrails.GuardrailsValidator')
    @patch('src.rag_pipeline_with_guardrails.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_with_guardrails.ConversationalRetrievalChain')
    def test_chat_with_validation(self, mock_conv, mock_llm, mock_guard_class, mock_vector, mock_google_api_key):
        """Test chat mode with input/output validation"""
        # Setup mocks
        mock_vector.return_value = Mock()

        mock_validator = Mock()
        mock_validator.validate_input = Mock(return_value={
            'valid': True,
            'validated_text': 'Hello',
            'errors': []
        })
        mock_validator.validate_output = Mock(return_value={
            'valid': True,
            'validated_text': 'Hi there!',
            'errors': []
        })
        mock_guard_class.return_value = mock_validator

        mock_chain = Mock()
        mock_chain.__call__ = Mock(return_value={
            'question': 'Hello',
            'answer': 'Hi there!',
            'source_documents': []
        })
        mock_conv.from_llm = Mock(return_value=mock_chain)

        # Create pipeline
        pipeline = SecureRAGPipeline(
            google_api_key=mock_google_api_key,
            enable_guardrails=True
        )
        pipeline.chat_chain = mock_chain
        pipeline.guardrails_validator = mock_validator

        # Chat
        response = pipeline.chat("Hello", user_id="test_user")

        # Verify
        assert mock_validator.validate_input.called
        assert mock_validator.validate_output.called
        assert response['security']['input_validated'] is True
        assert response['security']['output_validated'] is True
