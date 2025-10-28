"""
Pytest configuration and shared fixtures
Provides common test utilities and fixtures for all tests
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_articles():
    """Sample Wikipedia articles for testing"""
    return [
        {
            "id": "test_article_1_chunk_0",
            "title": "Machine Learning",
            "url": "https://en.wikipedia.org/wiki/Machine_Learning",
            "content": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
            "summary": "Machine learning is a method of data analysis.",
            "topic": "Machine Learning",
            "categories": ["Computer Science", "Artificial Intelligence"],
            "chunk_index": 0,
            "total_chunks": 1,
            "retrieved_at": "2024-01-01T00:00:00"
        },
        {
            "id": "test_article_2_chunk_0",
            "title": "Deep Learning",
            "url": "https://en.wikipedia.org/wiki/Deep_Learning",
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "summary": "Deep learning is based on neural networks.",
            "topic": "Deep Learning",
            "categories": ["Neural Networks", "Artificial Intelligence"],
            "chunk_index": 0,
            "total_chunks": 1,
            "retrieved_at": "2024-01-01T00:00:00"
        },
        {
            "id": "test_article_3_chunk_0",
            "title": "Natural Language Processing",
            "url": "https://en.wikipedia.org/wiki/Natural_Language_Processing",
            "content": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            "summary": "NLP deals with human-computer language interaction.",
            "topic": "NLP",
            "categories": ["Linguistics", "Computer Science"],
            "chunk_index": 0,
            "total_chunks": 1,
            "retrieved_at": "2024-01-01T00:00:00"
        }
    ]

@pytest.fixture(scope="session")
def sample_articles_file(test_data_dir, sample_articles):
    """Create a sample articles JSON file"""
    articles_file = os.path.join(test_data_dir, "test_articles.json")
    with open(articles_file, 'w', encoding='utf-8') as f:
        json.dump(sample_articles, f, indent=2)
    return articles_file

@pytest.fixture
def mock_google_api_key(monkeypatch):
    """Mock Google API key for testing"""
    api_key = "test_google_api_key_123"
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    return api_key

@pytest.fixture
def mock_openai_api_key(monkeypatch):
    """Mock OpenAI API key for testing"""
    api_key = "test_openai_api_key_123"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key

@pytest.fixture
def mock_wikipedia_page():
    """Mock Wikipedia page object"""
    page = Mock()
    page.title = "Test Article"
    page.url = "https://en.wikipedia.org/wiki/Test_Article"
    page.content = "This is test content for the article. " * 50
    page.summary = "This is a test summary."
    page.categories = ["Category1", "Category2", "Category3"]
    return page

@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model"""
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 384
    model.encode.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    return model

@pytest.fixture
def mock_gemini_model():
    """Mock Google Gemini model"""
    model = Mock()
    response = Mock()
    response.text = "This is a test response from Gemini."
    model.generate_content.return_value = response
    return model

@pytest.fixture
def temp_vector_store_dir(test_data_dir):
    """Create temporary vector store directory"""
    vector_dir = os.path.join(test_data_dir, "vector_store")
    os.makedirs(vector_dir, exist_ok=True)
    return vector_dir

@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "What is machine learning?"

@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    import numpy as np
    return np.random.rand(3, 384).astype('float32')

# Cleanup hooks
def pytest_sessionfinish(session, exitstatus):
    """Cleanup after all tests"""
    # Clean up any test artifacts
    pass

# Custom markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "api: Tests that require external API calls"
    )
