"""
Unit tests for RAG Pipeline with Gemini
Tests the complete RAG pipeline functionality
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from src.rag_pipeline_gemini import RAGPipeline, CustomFAISSRetriever


@pytest.mark.unit
class TestCustomFAISSRetriever:
    """Test cases for CustomFAISSRetriever"""

    @pytest.fixture
    def mock_vector_store_manager(self):
        """Create a mock VectorStoreManager"""
        manager = Mock()
        manager.search_documents.return_value = [
            {
                'content': 'Test content 1',
                'title': 'Test Title 1',
                'url': 'https://test.com/1',
                'topic': 'Test',
                'similarity_score': 0.95,
                'chunk_index': 0
            },
            {
                'content': 'Test content 2',
                'title': 'Test Title 2',
                'url': 'https://test.com/2',
                'topic': 'Test',
                'similarity_score': 0.85,
                'chunk_index': 0
            }
        ]
        return manager

    def test_retriever_initialization(self, mock_vector_store_manager):
        """Test retriever initialization"""
        retriever = CustomFAISSRetriever(
            vector_store_manager=mock_vector_store_manager,
            k=5
        )

        assert retriever.vector_store_manager == mock_vector_store_manager
        assert retriever.k == 5

    def test_get_relevant_documents(self, mock_vector_store_manager):
        """Test retrieving relevant documents"""
        retriever = CustomFAISSRetriever(
            vector_store_manager=mock_vector_store_manager,
            k=2
        )

        query = "test query"
        documents = retriever.get_relevant_documents(query)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == 'Test content 1'
        assert documents[0].metadata['title'] == 'Test Title 1'
        assert documents[0].metadata['similarity_score'] == 0.95

    def test_get_relevant_documents_metadata(self, mock_vector_store_manager):
        """Test that retrieved documents have correct metadata"""
        retriever = CustomFAISSRetriever(
            vector_store_manager=mock_vector_store_manager,
            k=2
        )

        documents = retriever.get_relevant_documents("query")

        for doc in documents:
            assert 'title' in doc.metadata
            assert 'url' in doc.metadata
            assert 'topic' in doc.metadata
            assert 'similarity_score' in doc.metadata
            assert 'source' in doc.metadata


@pytest.mark.unit
class TestRAGPipeline:
    """Test cases for RAGPipeline class"""

    @pytest.fixture
    def mock_google_api_key(self):
        """Mock Google API key"""
        return "test_google_api_key_123"

    @pytest.fixture
    def temp_pipeline_dir(self, test_data_dir):
        """Create temporary pipeline directory"""
        pipeline_dir = os.path.join(test_data_dir, "test_pipeline")
        os.makedirs(pipeline_dir, exist_ok=True)
        return pipeline_dir

    def test_pipeline_initialization(self, mock_google_api_key, temp_pipeline_dir):
        """Test RAG pipeline initialization"""
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir,
            model_name="gemini-1.5-flash",
            temperature=0.1
        )

        assert pipeline.google_api_key == mock_google_api_key
        assert pipeline.model_name == "gemini-1.5-flash"
        assert pipeline.temperature == 0.1
        assert pipeline.llm is None
        assert pipeline.retriever is None

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    @patch('google.generativeai.configure')
    def test_pipeline_initialize(self, mock_genai_config, mock_vector_manager,
                                 mock_llm, mock_google_api_key, temp_pipeline_dir,
                                 sample_articles_file):
        """Test pipeline initialization with components"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_instance.initialize_from_articles = Mock()
        mock_vector_manager.return_value = mock_vector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create pipeline
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir
        )

        # Initialize
        pipeline.initialize(articles_file=sample_articles_file)

        # Verify components initialized
        assert pipeline.llm is not None
        assert pipeline.retriever is not None
        assert pipeline.qa_chain is not None
        assert pipeline.chat_chain is not None
        assert pipeline.memory is not None

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    @patch('google.generativeai.configure')
    def test_ask_question(self, mock_genai_config, mock_vector_manager,
                         mock_llm, mock_google_api_key, temp_pipeline_dir,
                         sample_articles_file):
        """Test asking a question"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_instance.initialize_from_articles = Mock()
        mock_vector_manager.return_value = mock_vector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create and initialize pipeline
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir
        )
        pipeline.initialize(articles_file=sample_articles_file)

        # Mock QA chain response
        mock_response = {
            "result": "This is a test answer",
            "source_documents": [
                Document(
                    page_content="Test content",
                    metadata={
                        'title': 'Test',
                        'url': 'https://test.com',
                        'similarity_score': 0.9
                    }
                )
            ]
        }
        pipeline.qa_chain = Mock(return_value=mock_response)

        # Ask question
        response = pipeline.ask_question("What is machine learning?")

        # Verify response structure
        assert "question" in response
        assert "answer" in response
        assert "source_documents" in response
        assert "usage" in response
        assert "timestamp" in response
        assert response["answer"] == "This is a test answer"

    def test_ask_question_without_initialization(self, mock_google_api_key, temp_pipeline_dir):
        """Test asking question before initialization"""
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir
        )

        with pytest.raises(ValueError, match="Pipeline not initialized"):
            pipeline.ask_question("test question")

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    @patch('google.generativeai.configure')
    def test_chat(self, mock_genai_config, mock_vector_manager, mock_llm,
                  mock_google_api_key, temp_pipeline_dir, sample_articles_file):
        """Test chat functionality"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_instance.initialize_from_articles = Mock()
        mock_vector_manager.return_value = mock_vector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create and initialize pipeline
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir
        )
        pipeline.initialize(articles_file=sample_articles_file)

        # Mock chat chain response
        mock_response = {
            "answer": "This is a chat response",
            "source_documents": [
                Document(
                    page_content="Test content",
                    metadata={
                        'title': 'Test',
                        'url': 'https://test.com',
                        'similarity_score': 0.9
                    }
                )
            ]
        }
        pipeline.chat_chain = Mock(return_value=mock_response)

        # Send chat message
        response = pipeline.chat("Tell me about AI")

        # Verify response
        assert "message" in response
        assert "response" in response
        assert "source_documents" in response
        assert response["response"] == "This is a chat response"

    def test_chat_without_initialization(self, mock_google_api_key, temp_pipeline_dir):
        """Test chat before initialization"""
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir
        )

        with pytest.raises(ValueError, match="Pipeline not initialized"):
            pipeline.chat("test message")

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    @patch('google.generativeai.configure')
    def test_clear_memory(self, mock_genai_config, mock_vector_manager,
                         mock_llm, mock_google_api_key, temp_pipeline_dir,
                         sample_articles_file):
        """Test clearing conversation memory"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_instance.initialize_from_articles = Mock()
        mock_vector_manager.return_value = mock_vector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create and initialize pipeline
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir
        )
        pipeline.initialize(articles_file=sample_articles_file)

        # Clear memory
        pipeline.clear_memory()

        # Verify memory was cleared
        pipeline.memory.clear.assert_called_once()

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    @patch('google.generativeai.configure')
    def test_get_stats(self, mock_genai_config, mock_vector_manager,
                      mock_llm, mock_google_api_key, temp_pipeline_dir,
                      sample_articles_file):
        """Test getting pipeline statistics"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_instance.initialize_from_articles = Mock()
        mock_vector_stats = {
            'num_documents': 10,
            'num_vectors': 10,
            'model_name': 'test-model'
        }
        mock_vector_instance.vector_store.get_stats.return_value = mock_vector_stats
        mock_vector_manager.return_value = mock_vector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create and initialize pipeline
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir,
            model_name="gemini-1.5-flash",
            temperature=0.2
        )
        pipeline.initialize(articles_file=sample_articles_file)

        # Get stats
        stats = pipeline.get_stats()

        # Verify stats
        assert stats["model_name"] == "gemini-1.5-flash"
        assert stats["temperature"] == 0.2
        assert stats["initialized"] is True
        assert stats["provider"] == "Google Gemini"
        assert "memory_size" in stats

    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    def test_get_context_for_query(self, mock_vector_manager, mock_google_api_key,
                                   temp_pipeline_dir):
        """Test getting context for a query"""
        # Setup mock
        mock_vector_instance = Mock()
        mock_vector_instance.get_context_for_query.return_value = "Test context"
        mock_vector_manager.return_value = mock_vector_instance

        # Create pipeline
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir
        )
        pipeline.vector_store_manager = mock_vector_instance

        # Get context
        context = pipeline.get_context_for_query("test query", max_context_length=1000)

        assert context == "Test context"
        mock_vector_instance.get_context_for_query.assert_called_once_with("test query", 1000)

    def test_get_context_without_vector_store(self, mock_google_api_key, temp_pipeline_dir):
        """Test getting context before vector store initialization"""
        pipeline = RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_pipeline_dir
        )
        pipeline.vector_store_manager = None

        with pytest.raises(ValueError, match="Vector store not initialized"):
            pipeline.get_context_for_query("test")


@pytest.mark.unit
class TestRAGPipelineEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def pipeline(self, mock_google_api_key, temp_vector_store_dir):
        return RAGPipeline(
            google_api_key=mock_google_api_key,
            vector_store_dir=temp_vector_store_dir
        )

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    @patch('google.generativeai.configure')
    def test_ask_question_with_error(self, mock_genai_config, mock_vector_manager,
                                    mock_llm, pipeline, sample_articles_file):
        """Test handling of errors during question answering"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_instance.initialize_from_articles = Mock()
        mock_vector_manager.return_value = mock_vector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Initialize pipeline
        pipeline.initialize(articles_file=sample_articles_file)

        # Make QA chain raise an error
        pipeline.qa_chain = Mock(side_effect=Exception("API Error"))

        # Should raise exception
        with pytest.raises(Exception):
            pipeline.ask_question("test")

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    @patch('google.generativeai.configure')
    def test_empty_source_documents(self, mock_genai_config, mock_vector_manager,
                                    mock_llm, pipeline, sample_articles_file):
        """Test handling of responses with no source documents"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_instance.initialize_from_articles = Mock()
        mock_vector_manager.return_value = mock_vector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Initialize pipeline
        pipeline.initialize(articles_file=sample_articles_file)

        # Mock response with no source documents
        mock_response = {
            "result": "Answer without sources",
            "source_documents": []
        }
        pipeline.qa_chain = Mock(return_value=mock_response)

        # Ask question
        response = pipeline.ask_question("test")

        # Should handle empty source documents
        assert response["source_documents"] == []

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline_gemini.VectorStoreManager')
    @patch('google.generativeai.configure')
    def test_very_long_content_truncation(self, mock_genai_config, mock_vector_manager,
                                         mock_llm, pipeline, sample_articles_file):
        """Test that very long content is properly truncated"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector_instance.initialize_from_articles = Mock()
        mock_vector_manager.return_value = mock_vector_instance

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Initialize pipeline
        pipeline.initialize(articles_file=sample_articles_file)

        # Mock response with very long content
        long_content = "word " * 1000  # Very long content
        mock_response = {
            "result": "Answer",
            "source_documents": [
                Document(
                    page_content=long_content,
                    metadata={
                        'title': 'Test',
                        'url': 'https://test.com',
                        'similarity_score': 0.9
                    }
                )
            ]
        }
        pipeline.qa_chain = Mock(return_value=mock_response)

        # Ask question
        response = pipeline.ask_question("test")

        # Verify content is truncated
        returned_content = response["source_documents"][0]["content"]
        assert len(returned_content) <= 503  # 500 + "..."
