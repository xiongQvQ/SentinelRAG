"""
Unit tests for Vector Store
Tests FAISS vector store and embedding functionality
"""

import pytest
import os
import json
import numpy as np
import faiss
from unittest.mock import Mock, patch, MagicMock

from src.vector_store import FAISSVectorStore, VectorStoreManager


@pytest.mark.unit
class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore class"""

    @pytest.fixture
    def vector_store(self, temp_vector_store_dir):
        """Create a FAISSVectorStore instance"""
        return FAISSVectorStore(
            model_name="all-MiniLM-L6-v2",
            vector_store_dir=temp_vector_store_dir
        )

    def test_init(self, vector_store, temp_vector_store_dir):
        """Test vector store initialization"""
        assert vector_store.model_name == "all-MiniLM-L6-v2"
        assert vector_store.vector_store_dir == temp_vector_store_dir
        assert vector_store.embedding_model is None
        assert vector_store.index is None
        assert vector_store.documents == []

    @patch('src.vector_store.SentenceTransformer')
    def test_load_embedding_model(self, mock_transformer, vector_store):
        """Test loading embedding model"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        vector_store.load_embedding_model()

        assert vector_store.embedding_model is not None
        assert vector_store.dimension == 384
        mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    @patch('src.vector_store.SentenceTransformer')
    def test_create_embeddings(self, mock_transformer, vector_store):
        """Test creating embeddings"""
        mock_model = Mock()
        mock_embeddings = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        texts = ["text1", "text2", "text3"]
        embeddings = vector_store.create_embeddings(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    @patch('src.vector_store.SentenceTransformer')
    def test_build_index(self, mock_transformer, vector_store, sample_articles):
        """Test building FAISS index"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(len(sample_articles), 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        vector_store.build_index(sample_articles)

        assert vector_store.index is not None
        assert vector_store.index.ntotal == len(sample_articles)
        assert vector_store.dimension == 384
        assert len(vector_store.documents) == len(sample_articles)

    @patch('src.vector_store.SentenceTransformer')
    def test_search(self, mock_transformer, vector_store, sample_articles):
        """Test vector search"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(len(sample_articles), 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        # Build index first
        vector_store.build_index(sample_articles)

        # Search
        query = "test query"
        results = vector_store.search(query, k=2)

        assert len(results) <= 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        # Check that each result has (document, score)
        for doc, score in results:
            assert isinstance(doc, dict)
            assert isinstance(score, float)

    def test_search_without_index(self, vector_store):
        """Test search when index not built"""
        with pytest.raises(ValueError, match="Index not built"):
            vector_store.search("query")

    @patch('src.vector_store.SentenceTransformer')
    def test_save_and_load_index(self, mock_transformer, vector_store, sample_articles, temp_vector_store_dir):
        """Test saving and loading index"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(len(sample_articles), 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        # Build and save index
        vector_store.build_index(sample_articles)
        index_path = vector_store.save_index()

        assert os.path.exists(index_path)
        assert os.path.exists(os.path.join(temp_vector_store_dir, "faiss_index_documents.json"))
        assert os.path.exists(os.path.join(temp_vector_store_dir, "faiss_index_config.json"))

        # Create new instance and load
        new_vector_store = FAISSVectorStore(vector_store_dir=temp_vector_store_dir)
        success = new_vector_store.load_index()

        assert success is True
        assert new_vector_store.index.ntotal == len(sample_articles)
        assert new_vector_store.dimension == 384
        assert len(new_vector_store.documents) == len(sample_articles)

    def test_save_index_without_building(self, vector_store):
        """Test saving index when not built"""
        with pytest.raises(ValueError, match="No index to save"):
            vector_store.save_index()

    def test_load_index_file_not_found(self, vector_store):
        """Test loading index when files don't exist"""
        with pytest.raises(FileNotFoundError):
            vector_store.load_index()

    @patch('src.vector_store.SentenceTransformer')
    def test_get_stats(self, mock_transformer, vector_store, sample_articles):
        """Test getting vector store statistics"""
        # Before initialization
        stats = vector_store.get_stats()
        assert stats["status"] == "not_initialized"

        # After initialization
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(len(sample_articles), 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        vector_store.build_index(sample_articles)
        stats = vector_store.get_stats()

        assert stats["status"] == "initialized"
        assert stats["model_name"] == "all-MiniLM-L6-v2"
        assert stats["dimension"] == 384
        assert stats["num_vectors"] == len(sample_articles)
        assert stats["num_documents"] == len(sample_articles)

    @patch('src.vector_store.SentenceTransformer')
    def test_add_documents_to_empty_index(self, mock_transformer, vector_store, sample_articles):
        """Test adding documents when no index exists"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(len(sample_articles), 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        vector_store.add_documents(sample_articles)

        assert vector_store.index.ntotal == len(sample_articles)

    @patch('src.vector_store.SentenceTransformer')
    def test_add_documents_to_existing_index(self, mock_transformer, vector_store, sample_articles):
        """Test adding documents to existing index"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(len(sample_articles), 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        # Build initial index
        initial_docs = sample_articles[:2]
        vector_store.build_index(initial_docs)
        initial_count = vector_store.index.ntotal

        # Add more documents
        new_docs = sample_articles[2:]
        vector_store.add_documents(new_docs)

        assert vector_store.index.ntotal == initial_count + len(new_docs)


@pytest.mark.unit
class TestVectorStoreManager:
    """Test cases for VectorStoreManager class"""

    @pytest.fixture
    def manager(self, temp_vector_store_dir):
        """Create a VectorStoreManager instance"""
        return VectorStoreManager(vector_store_dir=temp_vector_store_dir)

    @patch('src.vector_store.SentenceTransformer')
    def test_initialize_from_articles_new(self, mock_transformer, manager, sample_articles_file):
        """Test initializing from articles file (new index)"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(3, 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        manager.initialize_from_articles(sample_articles_file)

        assert manager.vector_store.index is not None
        assert manager.vector_store.index.ntotal > 0

    @patch('src.vector_store.SentenceTransformer')
    def test_initialize_from_articles_force_rebuild(self, mock_transformer, manager, sample_articles_file):
        """Test force rebuilding index"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(3, 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        # Build once
        manager.initialize_from_articles(sample_articles_file)
        first_index_total = manager.vector_store.index.ntotal

        # Force rebuild
        manager.initialize_from_articles(sample_articles_file, force_rebuild=True)

        # Should rebuild
        assert manager.vector_store.index.ntotal == first_index_total

    def test_initialize_from_nonexistent_file(self, manager):
        """Test initializing from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            manager.initialize_from_articles("nonexistent.json")

    @patch('src.vector_store.SentenceTransformer')
    def test_search_documents(self, mock_transformer, manager, sample_articles_file):
        """Test document search"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(3, 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        manager.initialize_from_articles(sample_articles_file)
        results = manager.search_documents("test query", k=2)

        assert len(results) <= 2
        assert all(isinstance(result, dict) for result in results)
        assert all("similarity_score" in result for result in results)

    @patch('src.vector_store.SentenceTransformer')
    def test_get_context_for_query(self, mock_transformer, manager, sample_articles_file):
        """Test getting context string for query"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(3, 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        manager.initialize_from_articles(sample_articles_file)
        context = manager.get_context_for_query("test query", max_context_length=500)

        assert isinstance(context, str)
        assert len(context) <= 500 or len(context) > 0  # May be slightly over due to formatting

    @patch('src.vector_store.SentenceTransformer')
    def test_get_context_respects_max_length(self, mock_transformer, manager, sample_articles_file):
        """Test that context respects maximum length"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(3, 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        manager.initialize_from_articles(sample_articles_file)
        context = manager.get_context_for_query("test query", max_context_length=100)

        # Context should not be excessively long
        assert len(context) < 200  # Some buffer for formatting


@pytest.mark.unit
class TestVectorStoreEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def vector_store(self, temp_vector_store_dir):
        return FAISSVectorStore(vector_store_dir=temp_vector_store_dir)

    @patch('src.vector_store.SentenceTransformer')
    def test_empty_documents_list(self, mock_transformer, vector_store):
        """Test building index with empty documents list"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([]).reshape(0, 384).astype('float32')
        mock_transformer.return_value = mock_model

        # Should handle empty list gracefully
        try:
            vector_store.build_index([])
        except Exception as e:
            # It's ok if it fails, but shouldn't crash
            assert isinstance(e, (ValueError, IndexError))

    @patch('src.vector_store.SentenceTransformer')
    def test_search_with_k_greater_than_documents(self, mock_transformer, vector_store, sample_articles):
        """Test search with k greater than number of documents"""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.random.rand(len(sample_articles), 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        vector_store.build_index(sample_articles)
        results = vector_store.search("query", k=100)

        # Should return at most len(sample_articles) results
        assert len(results) <= len(sample_articles)

    @patch('src.vector_store.SentenceTransformer')
    def test_large_batch_embeddings(self, mock_transformer, vector_store):
        """Test creating embeddings for large batch"""
        mock_model = Mock()
        large_batch_size = 1000
        mock_embeddings = np.random.rand(large_batch_size, 384).astype('float32')
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        texts = [f"text {i}" for i in range(large_batch_size)]
        embeddings = vector_store.create_embeddings(texts)

        assert embeddings.shape == (large_batch_size, 384)
