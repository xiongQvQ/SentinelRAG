"""
Integration tests for RAG System
Tests integration between data collector, vector store, and RAG pipeline
"""

import pytest
import os
import json
import numpy as np
from unittest.mock import Mock, patch

from src.data_collector import WikipediaDataCollector
from src.vector_store import VectorStoreManager
from src.rag_pipeline_gemini import RAGPipeline


@pytest.mark.integration
class TestDataCollectorVectorStoreIntegration:
    """Test integration between data collector and vector store"""

    @pytest.fixture
    def collector(self, test_data_dir):
        return WikipediaDataCollector(data_dir=test_data_dir)

    @pytest.fixture
    def vector_manager(self, temp_vector_store_dir):
        return VectorStoreManager(vector_store_dir=temp_vector_store_dir)

    @patch('wikipedia.search')
    @patch('wikipedia.page')
    @patch('src.vector_store.SentenceTransformer')
    def test_collect_and_index_flow(self, mock_transformer, mock_page, mock_search,
                                    collector, vector_manager):
        """Test complete flow from collection to indexing"""
        # Mock Wikipedia search
        mock_search.return_value = ["Machine Learning", "AI"]

        # Mock Wikipedia page
        mock_wiki_page = Mock()
        mock_wiki_page.title = "Machine Learning"
        mock_wiki_page.url = "https://en.wikipedia.org/wiki/Machine_Learning"
        mock_wiki_page.content = "Machine learning is a subset of AI. " * 50
        mock_wiki_page.summary = "ML is part of AI"
        mock_wiki_page.categories = ["Computer Science", "AI"]
        mock_page.return_value = mock_wiki_page

        # Mock sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(10, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Collect articles
        topics = ["Machine Learning"]
        articles = collector.collect_articles_by_topics(topics, articles_per_topic=2)
        assert len(articles) > 0

        # Preprocess
        processed = collector.preprocess_content(articles)
        assert len(processed) > 0

        # Save
        articles_file = collector.save_articles(processed, "test_articles.json")
        assert os.path.exists(articles_file)

        # Initialize vector store from articles
        vector_manager.initialize_from_articles(articles_file)

        # Verify vector store has documents
        stats = vector_manager.vector_store.get_stats()
        assert stats["status"] == "initialized"
        assert stats["num_documents"] > 0

    @patch('src.vector_store.SentenceTransformer')
    def test_search_after_indexing(self, mock_transformer, vector_manager,
                                   sample_articles_file):
        """Test searching after indexing documents"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Initialize from existing articles
        vector_manager.initialize_from_articles(sample_articles_file)

        # Search
        results = vector_manager.search_documents("machine learning", k=2)

        assert len(results) > 0
        assert all('similarity_score' in r for r in results)

    @patch('src.vector_store.SentenceTransformer')
    def test_save_and_reload_vector_store(self, mock_transformer, vector_manager,
                                         sample_articles_file):
        """Test saving and reloading vector store"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Build and save
        vector_manager.initialize_from_articles(sample_articles_file)
        original_count = vector_manager.vector_store.index.ntotal

        # Create new manager and load
        new_manager = VectorStoreManager(vector_store_dir=vector_manager.vector_store_dir)
        new_manager.initialize_from_articles(sample_articles_file, force_rebuild=False)

        # Verify loaded correctly
        assert new_manager.vector_store.index.ntotal == original_count


@pytest.mark.integration
class TestVectorStoreRAGPipelineIntegration:
    """Test integration between vector store and RAG pipeline"""

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.vector_store.SentenceTransformer')
    @patch('google.generativeai.configure')
    def test_pipeline_with_vector_store(self, mock_genai_config, mock_transformer,
                                       mock_llm, temp_vector_store_dir,
                                       sample_articles_file):
        """Test RAG pipeline with vector store integration"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create pipeline
        pipeline = RAGPipeline(
            google_api_key="test_key",
            vector_store_dir=temp_vector_store_dir
        )

        # Initialize
        pipeline.initialize(articles_file=sample_articles_file)

        # Verify components are connected
        assert pipeline.vector_store_manager is not None
        assert pipeline.retriever is not None
        assert pipeline.retriever.vector_store_manager == pipeline.vector_store_manager

    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('src.vector_store.SentenceTransformer')
    @patch('google.generativeai.configure')
    def test_retrieval_in_pipeline(self, mock_genai_config, mock_transformer,
                                   mock_llm, temp_vector_store_dir,
                                   sample_articles_file):
        """Test document retrieval within pipeline"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create and initialize pipeline
        pipeline = RAGPipeline(
            google_api_key="test_key",
            vector_store_dir=temp_vector_store_dir
        )
        pipeline.initialize(articles_file=sample_articles_file)

        # Test retrieval
        documents = pipeline.retriever.get_relevant_documents("machine learning")

        assert len(documents) > 0
        assert all(hasattr(doc, 'page_content') for doc in documents)
        assert all(hasattr(doc, 'metadata') for doc in documents)


@pytest.mark.integration
class TestFullRAGWorkflow:
    """Test complete RAG workflow integration"""

    @patch('wikipedia.search')
    @patch('wikipedia.page')
    @patch('src.vector_store.SentenceTransformer')
    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('google.generativeai.configure')
    def test_end_to_end_workflow(self, mock_genai_config, mock_llm, mock_transformer,
                                 mock_page, mock_search, test_data_dir):
        """Test complete workflow from data collection to Q&A"""
        # Mock Wikipedia
        mock_search.return_value = ["Test Article"]
        mock_wiki_page = Mock()
        mock_wiki_page.title = "Test Article"
        mock_wiki_page.url = "https://test.com"
        mock_wiki_page.content = "This is test content about machine learning. " * 20
        mock_wiki_page.summary = "Test summary"
        mock_wiki_page.categories = ["Test"]
        mock_page.return_value = mock_wiki_page

        # Mock embeddings
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        def mock_encode(texts, show_progress_bar=False):
            return np.random.rand(len(texts), 384).astype('float32')
        mock_model.encode = mock_encode
        mock_transformer.return_value = mock_model

        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Step 1: Collect data
        collector = WikipediaDataCollector(data_dir=test_data_dir)
        articles = collector.collect_articles_by_topics(["Test"], articles_per_topic=1)
        processed = collector.preprocess_content(articles)
        articles_file = collector.save_articles(processed, "workflow_test.json")

        # Step 2: Initialize vector store
        vector_dir = os.path.join(test_data_dir, "vector_store")
        os.makedirs(vector_dir, exist_ok=True)
        vector_manager = VectorStoreManager(vector_store_dir=vector_dir)
        vector_manager.initialize_from_articles(articles_file)

        # Step 3: Initialize RAG pipeline
        pipeline = RAGPipeline(
            google_api_key="test_key",
            vector_store_dir=vector_dir
        )
        pipeline.initialize(articles_file=articles_file)

        # Step 4: Query (with mocked response)
        from langchain.schema import Document
        mock_qa_response = {
            "result": "Machine learning is a subset of AI",
            "source_documents": [
                Document(
                    page_content="Test content",
                    metadata={'title': 'Test', 'url': 'https://test.com', 'similarity_score': 0.9}
                )
            ]
        }
        pipeline.qa_chain = Mock(return_value=mock_qa_response)

        response = pipeline.ask_question("What is machine learning?")

        # Verify complete workflow
        assert response is not None
        assert "answer" in response
        assert "source_documents" in response

    @patch('src.vector_store.SentenceTransformer')
    def test_multiple_queries_same_session(self, mock_transformer, sample_articles_file,
                                          temp_vector_store_dir):
        """Test multiple queries in same session"""
        # Mock embeddings
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Initialize vector store
        vector_manager = VectorStoreManager(vector_store_dir=temp_vector_store_dir)
        vector_manager.initialize_from_articles(sample_articles_file)

        # Multiple queries
        queries = ["machine learning", "deep learning", "AI"]
        for query in queries:
            results = vector_manager.search_documents(query, k=2)
            assert len(results) > 0

    @patch('src.vector_store.SentenceTransformer')
    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('google.generativeai.configure')
    def test_conversation_memory_integration(self, mock_genai_config, mock_llm,
                                             mock_transformer, sample_articles_file,
                                             temp_vector_store_dir):
        """Test conversation memory across multiple interactions"""
        # Mock embeddings
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Create pipeline
        pipeline = RAGPipeline(
            google_api_key="test_key",
            vector_store_dir=temp_vector_store_dir
        )
        pipeline.initialize(articles_file=sample_articles_file)

        # Check memory is initialized
        assert pipeline.memory is not None

        # Verify memory window size
        assert pipeline.memory.k == 5  # Should remember last 5 exchanges

    @pytest.mark.slow
    @patch('wikipedia.search')
    @patch('wikipedia.page')
    @patch('src.vector_store.SentenceTransformer')
    def test_large_dataset_integration(self, mock_transformer, mock_page, mock_search,
                                      test_data_dir):
        """Test integration with larger dataset"""
        # Mock Wikipedia to return multiple results
        mock_search.return_value = [f"Article {i}" for i in range(10)]

        mock_wiki_page = Mock()
        mock_wiki_page.title = "Test Article"
        mock_wiki_page.url = "https://test.com"
        mock_wiki_page.content = "Content " * 100
        mock_wiki_page.summary = "Summary"
        mock_wiki_page.categories = ["Test"]
        mock_page.return_value = mock_wiki_page

        # Mock embeddings
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        def mock_encode(texts, show_progress_bar=False):
            return np.random.rand(len(texts), 384).astype('float32')
        mock_model.encode = mock_encode
        mock_transformer.return_value = mock_model

        # Collect larger dataset
        collector = WikipediaDataCollector(data_dir=test_data_dir)
        articles = collector.collect_articles_by_topics(["Test"], articles_per_topic=10)
        processed = collector.preprocess_content(articles)

        # Should handle large number of chunks
        assert len(processed) >= 10

        # Save and index
        articles_file = collector.save_articles(processed, "large_test.json")
        vector_dir = os.path.join(test_data_dir, "large_vector_store")
        os.makedirs(vector_dir, exist_ok=True)

        vector_manager = VectorStoreManager(vector_store_dir=vector_dir)
        vector_manager.initialize_from_articles(articles_file)

        # Verify all indexed
        stats = vector_manager.vector_store.get_stats()
        assert stats["num_documents"] >= 10
