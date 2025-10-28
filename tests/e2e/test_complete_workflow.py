"""
End-to-end tests for complete RAG system workflow
Tests the entire system from user perspective
"""

import pytest
import os
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.data_collector import WikipediaDataCollector
from src.vector_store import VectorStoreManager
from src.rag_pipeline_gemini import RAGPipeline


@pytest.mark.e2e
class TestCompleteSystemWorkflow:
    """End-to-end tests for complete system workflow"""

    @pytest.fixture
    def system_setup(self, test_data_dir):
        """Setup complete system environment"""
        data_dir = os.path.join(test_data_dir, "e2e_data")
        vector_dir = os.path.join(test_data_dir, "e2e_vector")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)

        return {
            'data_dir': data_dir,
            'vector_dir': vector_dir,
            'api_key': 'test_api_key_e2e'
        }

    @patch('wikipedia.search')
    @patch('wikipedia.page')
    @patch('src.vector_store.SentenceTransformer')
    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('google.generativeai.configure')
    def test_complete_user_workflow(self, mock_genai_config, mock_llm, mock_transformer,
                                   mock_page, mock_search, system_setup):
        """Test complete workflow from user perspective"""
        # Mock Wikipedia API
        mock_search.return_value = ["Artificial Intelligence", "Machine Learning"]
        mock_wiki_page = Mock()
        mock_wiki_page.title = "Artificial Intelligence"
        mock_wiki_page.url = "https://en.wikipedia.org/wiki/Artificial_Intelligence"
        mock_wiki_page.content = "Artificial intelligence (AI) is intelligence demonstrated by machines. " * 30
        mock_wiki_page.summary = "AI is machine intelligence"
        mock_wiki_page.categories = ["Computer Science", "AI"]
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

        # USER STORY: User wants to build a Q&A system about AI

        # Step 1: Collect data about AI topics
        collector = WikipediaDataCollector(data_dir=system_setup['data_dir'])
        topics = ["Artificial Intelligence", "Machine Learning"]
        raw_articles = collector.collect_articles_by_topics(topics, articles_per_topic=2)

        # Verify data collection
        assert len(raw_articles) > 0
        assert all('content' in article for article in raw_articles)

        # Step 2: Preprocess articles into chunks
        processed_articles = collector.preprocess_content(raw_articles)

        # Verify preprocessing
        assert len(processed_articles) >= len(raw_articles)
        assert all('chunk_index' in chunk for chunk in processed_articles)

        # Step 3: Save processed data
        articles_file = collector.save_articles(processed_articles, "ai_articles.json")
        assert os.path.exists(articles_file)

        # Step 4: Build vector index
        vector_manager = VectorStoreManager(vector_store_dir=system_setup['vector_dir'])
        vector_manager.initialize_from_articles(articles_file)

        # Verify indexing
        stats = vector_manager.vector_store.get_stats()
        assert stats['status'] == 'initialized'
        assert stats['num_documents'] > 0

        # Step 5: Initialize RAG pipeline
        pipeline = RAGPipeline(
            google_api_key=system_setup['api_key'],
            vector_store_dir=system_setup['vector_dir']
        )
        pipeline.initialize(articles_file=articles_file)

        # Verify pipeline ready
        assert pipeline.llm is not None
        assert pipeline.retriever is not None
        assert pipeline.qa_chain is not None

        # Step 6: User asks questions
        from langchain.schema import Document
        test_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the applications of AI?"
        ]

        for i, question in enumerate(test_questions):
            # Mock LLM response
            mock_qa_response = {
                "result": f"Answer to question {i+1}",
                "source_documents": [
                    Document(
                        page_content="AI is machine intelligence",
                        metadata={
                            'title': 'Artificial Intelligence',
                            'url': 'https://en.wikipedia.org/wiki/AI',
                            'similarity_score': 0.9
                        }
                    )
                ]
            }
            pipeline.qa_chain = Mock(return_value=mock_qa_response)

            response = pipeline.ask_question(question)

            # Verify response structure
            assert 'question' in response
            assert 'answer' in response
            assert 'source_documents' in response
            assert 'usage' in response
            assert 'timestamp' in response

            # Verify source citations
            assert len(response['source_documents']) > 0
            assert all('title' in doc for doc in response['source_documents'])

        # Step 7: Test conversation memory
        assert pipeline.memory is not None

        # Step 8: Clear memory
        pipeline.clear_memory()

        # End-to-end workflow complete

    @patch('src.vector_store.SentenceTransformer')
    def test_search_quality_workflow(self, mock_transformer, sample_articles_file, test_data_dir):
        """Test search quality in end-to-end workflow"""
        # Mock embeddings with controlled similarity
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Create embeddings that will produce predictable similarity
        def mock_encode(texts, show_progress_bar=False):
            embeddings = []
            for i in range(len(texts)):
                # Different embeddings for different texts
                emb = np.zeros(384)
                emb[i % 384] = 1.0
                embeddings.append(emb)
            return np.array(embeddings, dtype='float32')

        mock_model.encode = mock_encode
        mock_transformer.return_value = mock_model

        # Setup vector store
        vector_dir = os.path.join(test_data_dir, "search_quality_vector")
        os.makedirs(vector_dir, exist_ok=True)

        vector_manager = VectorStoreManager(vector_store_dir=vector_dir)
        vector_manager.initialize_from_articles(sample_articles_file)

        # Test search returns results
        results = vector_manager.search_documents("machine learning", k=3)

        assert len(results) > 0
        assert len(results) <= 3

        # Verify all results have required fields
        for result in results:
            assert 'content' in result
            assert 'title' in result
            assert 'similarity_score' in result
            assert 'url' in result

    @patch('wikipedia.search')
    @patch('wikipedia.page')
    @patch('src.vector_store.SentenceTransformer')
    def test_incremental_data_addition(self, mock_transformer, mock_page, mock_search, test_data_dir):
        """Test adding data incrementally to existing system"""
        # Mock Wikipedia
        mock_search.return_value = ["Article 1"]
        mock_wiki_page = Mock()
        mock_wiki_page.title = "Test Article"
        mock_wiki_page.url = "https://test.com"
        mock_wiki_page.content = "Content " * 50
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

        # Initial setup
        data_dir = os.path.join(test_data_dir, "incremental_data")
        vector_dir = os.path.join(test_data_dir, "incremental_vector")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)

        collector = WikipediaDataCollector(data_dir=data_dir)

        # Add initial data
        articles_1 = collector.collect_articles_by_topics(["Topic1"], articles_per_topic=1)
        processed_1 = collector.preprocess_content(articles_1)
        file_1 = collector.save_articles(processed_1, "batch1.json")

        vector_manager = VectorStoreManager(vector_store_dir=vector_dir)
        vector_manager.initialize_from_articles(file_1)
        initial_count = vector_manager.vector_store.index.ntotal

        # Add more data
        articles_2 = collector.collect_articles_by_topics(["Topic2"], articles_per_topic=1)
        processed_2 = collector.preprocess_content(articles_2)

        # Add to existing index
        vector_manager.vector_store.add_documents(processed_2)
        new_count = vector_manager.vector_store.index.ntotal

        # Verify incremental addition
        assert new_count > initial_count

    @pytest.mark.slow
    @patch('wikipedia.search')
    @patch('wikipedia.page')
    @patch('src.vector_store.SentenceTransformer')
    @patch('src.rag_pipeline_gemini.ChatGoogleGenerativeAI')
    @patch('google.generativeai.configure')
    def test_system_persistence(self, mock_genai_config, mock_llm, mock_transformer,
                                mock_page, mock_search, test_data_dir):
        """Test system state persistence across restarts"""
        # Mock all external dependencies
        mock_search.return_value = ["Test"]
        mock_wiki_page = Mock()
        mock_wiki_page.title = "Test"
        mock_wiki_page.url = "https://test.com"
        mock_wiki_page.content = "Content " * 50
        mock_wiki_page.summary = "Summary"
        mock_wiki_page.categories = ["Test"]
        mock_page.return_value = mock_wiki_page

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        def mock_encode(texts, show_progress_bar=False):
            return np.random.rand(len(texts), 384).astype('float32')
        mock_model.encode = mock_encode
        mock_transformer.return_value = mock_model

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Setup directories
        data_dir = os.path.join(test_data_dir, "persistence_data")
        vector_dir = os.path.join(test_data_dir, "persistence_vector")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)

        # Session 1: Build system
        collector = WikipediaDataCollector(data_dir=data_dir)
        articles = collector.collect_articles_by_topics(["Test"], articles_per_topic=1)
        processed = collector.preprocess_content(articles)
        articles_file = collector.save_articles(processed, "persist_test.json")

        vector_manager_1 = VectorStoreManager(vector_store_dir=vector_dir)
        vector_manager_1.initialize_from_articles(articles_file)
        count_1 = vector_manager_1.vector_store.index.ntotal

        # Save index
        vector_manager_1.vector_store.save_index()

        # Session 2: Reload system
        vector_manager_2 = VectorStoreManager(vector_store_dir=vector_dir)
        vector_manager_2.initialize_from_articles(articles_file, force_rebuild=False)
        count_2 = vector_manager_2.vector_store.index.ntotal

        # Verify persistence
        assert count_1 == count_2

        # Both sessions should produce same search results
        query = "test query"
        results_1 = vector_manager_1.search_documents(query, k=2)
        results_2 = vector_manager_2.search_documents(query, k=2)

        assert len(results_1) == len(results_2)

    def test_error_recovery_workflow(self, sample_articles_file, test_data_dir):
        """Test system recovery from various error conditions"""
        vector_dir = os.path.join(test_data_dir, "error_recovery_vector")
        os.makedirs(vector_dir, exist_ok=True)

        # Test 1: Handle missing articles file gracefully
        vector_manager = VectorStoreManager(vector_store_dir=vector_dir)
        with pytest.raises(FileNotFoundError):
            vector_manager.initialize_from_articles("nonexistent.json")

        # Test 2: Handle missing vector index gracefully
        pipeline = RAGPipeline(
            google_api_key="test_key",
            vector_store_dir=vector_dir
        )
        # Should raise error if trying to use before initialization
        with pytest.raises(ValueError):
            pipeline.ask_question("test")

    @patch('src.vector_store.SentenceTransformer')
    def test_performance_metrics(self, mock_transformer, sample_articles_file, test_data_dir):
        """Test that system provides performance metrics"""
        # Mock embeddings
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Setup system
        vector_dir = os.path.join(test_data_dir, "metrics_vector")
        os.makedirs(vector_dir, exist_ok=True)

        vector_manager = VectorStoreManager(vector_store_dir=vector_dir)
        vector_manager.initialize_from_articles(sample_articles_file)

        # Get stats
        stats = vector_manager.vector_store.get_stats()

        # Verify metrics available
        assert 'status' in stats
        assert 'num_documents' in stats
        assert 'num_vectors' in stats
        assert 'model_name' in stats


@pytest.mark.e2e
@pytest.mark.slow
class TestSystemScalability:
    """Test system scalability and limits"""

    @patch('wikipedia.search')
    @patch('wikipedia.page')
    @patch('src.vector_store.SentenceTransformer')
    def test_large_document_handling(self, mock_transformer, mock_page, mock_search, test_data_dir):
        """Test handling of large documents"""
        # Mock very large document
        mock_search.return_value = ["Large Doc"]
        mock_wiki_page = Mock()
        mock_wiki_page.title = "Large Document"
        mock_wiki_page.url = "https://test.com"
        mock_wiki_page.content = "word " * 10000  # Very large content
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

        # Process large document
        data_dir = os.path.join(test_data_dir, "large_doc_data")
        os.makedirs(data_dir, exist_ok=True)

        collector = WikipediaDataCollector(data_dir=data_dir)
        articles = collector.collect_articles_by_topics(["Test"], articles_per_topic=1)
        processed = collector.preprocess_content(articles)

        # Verify chunking
        assert len(processed) > 1  # Should be split into multiple chunks
        assert all(len(chunk['content']) <= 1200 for chunk in processed)  # Reasonable chunk size

    def test_concurrent_searches(self, sample_articles_file, test_data_dir):
        """Test multiple concurrent search operations"""
        # This is a simplified test; real concurrency would need threading
        with patch('src.vector_store.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
            mock_transformer.return_value = mock_model

            vector_dir = os.path.join(test_data_dir, "concurrent_vector")
            os.makedirs(vector_dir, exist_ok=True)

            vector_manager = VectorStoreManager(vector_store_dir=vector_dir)
            vector_manager.initialize_from_articles(sample_articles_file)

            # Simulate multiple searches
            queries = [f"query {i}" for i in range(10)]
            results_list = []

            for query in queries:
                results = vector_manager.search_documents(query, k=2)
                results_list.append(results)

            # All searches should succeed
            assert len(results_list) == len(queries)
            assert all(len(r) > 0 for r in results_list)
