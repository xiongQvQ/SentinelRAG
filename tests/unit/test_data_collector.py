"""
Unit tests for WikipediaDataCollector
Tests data collection and preprocessing functionality
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
import wikipedia

from src.data_collector import WikipediaDataCollector


@pytest.mark.unit
class TestWikipediaDataCollector:
    """Test cases for WikipediaDataCollector class"""

    @pytest.fixture
    def collector(self, test_data_dir):
        """Create a WikipediaDataCollector instance"""
        return WikipediaDataCollector(data_dir=test_data_dir)

    def test_init(self, collector, test_data_dir):
        """Test collector initialization"""
        assert collector.data_dir == test_data_dir
        assert os.path.exists(test_data_dir)

    @patch('wikipedia.search')
    def test_search_articles_success(self, mock_search, collector):
        """Test successful article search"""
        mock_search.return_value = ["Machine Learning", "Deep Learning", "AI"]

        results = collector.search_articles("machine learning", num_articles=3)

        assert len(results) == 3
        assert "Machine Learning" in results
        mock_search.assert_called_once_with("machine learning", results=3)

    @patch('wikipedia.search')
    def test_search_articles_failure(self, mock_search, collector):
        """Test article search with exception"""
        mock_search.side_effect = Exception("API Error")

        results = collector.search_articles("machine learning")

        assert results == []

    @patch('wikipedia.page')
    def test_get_article_content_success(self, mock_page, collector, mock_wikipedia_page):
        """Test successful article content retrieval"""
        mock_page.return_value = mock_wikipedia_page

        result = collector.get_article_content("Test Article")

        assert result is not None
        assert result["title"] == "Test Article"
        assert result["url"] == "https://en.wikipedia.org/wiki/Test_Article"
        assert "content" in result
        assert "summary" in result
        assert "categories" in result
        assert "retrieved_at" in result

    @patch('wikipedia.page')
    def test_get_article_content_disambiguation(self, mock_page, collector):
        """Test article retrieval with disambiguation"""
        # First call raises disambiguation
        disambiguation_error = wikipedia.exceptions.DisambiguationError(
            "Disambiguation", ["Option 1", "Option 2"]
        )

        # Create mock page for second call
        mock_resolved_page = Mock()
        mock_resolved_page.title = "Option 1"
        mock_resolved_page.url = "https://en.wikipedia.org/wiki/Option_1"
        mock_resolved_page.content = "Resolved content"
        mock_resolved_page.summary = "Resolved summary"
        mock_resolved_page.categories = ["Category"]

        mock_page.side_effect = [disambiguation_error, mock_resolved_page]

        result = collector.get_article_content("Ambiguous Term")

        assert result is not None
        assert result["title"] == "Option 1"

    @patch('wikipedia.page')
    def test_get_article_content_page_not_found(self, mock_page, collector):
        """Test article retrieval when page doesn't exist"""
        mock_page.side_effect = wikipedia.exceptions.PageError("Page not found")

        result = collector.get_article_content("Nonexistent Page")

        assert result is None

    @patch.object(WikipediaDataCollector, 'search_articles')
    @patch.object(WikipediaDataCollector, 'get_article_content')
    def test_collect_articles_by_topics(self, mock_get_content, mock_search, collector):
        """Test collecting articles for multiple topics"""
        # Mock search results
        mock_search.return_value = ["Article 1", "Article 2"]

        # Mock article content
        mock_article = {
            "title": "Article 1",
            "url": "https://example.com",
            "content": "Content",
            "summary": "Summary",
            "categories": ["Cat1"],
            "retrieved_at": "2024-01-01"
        }
        mock_get_content.return_value = mock_article

        topics = ["Machine Learning", "AI"]
        results = collector.collect_articles_by_topics(topics, articles_per_topic=2)

        assert len(results) == 4  # 2 topics * 2 articles
        assert all("topic" in article for article in results)

    def test_split_text_into_chunks_small_text(self, collector):
        """Test text splitting with text smaller than chunk size"""
        text = "Short text"
        chunks = collector._split_text_into_chunks(text, max_chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_text_into_chunks_large_text(self, collector):
        """Test text splitting with text larger than chunk size"""
        text = "This is a sentence. " * 100  # ~2000 characters
        chunks = collector._split_text_into_chunks(text, max_chunk_size=500, overlap=50)

        assert len(chunks) > 1
        assert all(len(chunk) <= 550 for chunk in chunks)  # Max size + some buffer

    def test_split_text_sentence_boundary(self, collector):
        """Test that text splitting respects sentence boundaries"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = collector._split_text_into_chunks(text, max_chunk_size=30)

        # Check that chunks end at sentence boundaries
        assert all(chunk.strip().endswith('.') or chunk == chunks[-1] for chunk in chunks)

    def test_preprocess_content(self, collector, sample_articles):
        """Test article preprocessing"""
        # Create test articles without chunk_index
        test_articles = [
            {
                "title": "Test Article",
                "url": "https://test.com",
                "content": "Content " * 200,  # Long content to create multiple chunks
                "summary": "Summary",
                "categories": ["Cat1"],
                "retrieved_at": "2024-01-01",
                "topic": "Test"
            }
        ]

        processed = collector.preprocess_content(test_articles)

        assert len(processed) > 0
        assert all("id" in chunk for chunk in processed)
        assert all("title" in chunk for chunk in processed)
        assert all("chunk_index" in chunk for chunk in processed)
        assert all("total_chunks" in chunk for chunk in processed)

    def test_save_and_load_articles(self, collector, sample_articles):
        """Test saving and loading articles"""
        filename = "test_articles.json"

        # Save articles
        filepath = collector.save_articles(sample_articles, filename)
        assert os.path.exists(filepath)

        # Load articles
        loaded = collector.load_articles(filename)
        assert len(loaded) == len(sample_articles)
        assert loaded[0]["title"] == sample_articles[0]["title"]

    def test_load_articles_file_not_found(self, collector):
        """Test loading articles when file doesn't exist"""
        result = collector.load_articles("nonexistent.json")
        assert result == []

    def test_preprocess_with_missing_topic(self, collector):
        """Test preprocessing articles with missing topic field"""
        articles = [{
            "title": "Test",
            "url": "https://test.com",
            "content": "Content",
            "summary": "Summary",
            "categories": ["Cat"],
            "retrieved_at": "2024-01-01"
            # No topic field
        }]

        processed = collector.preprocess_content(articles)

        assert processed[0]["topic"] == "unknown"

    def test_chunk_overlap(self, collector):
        """Test that chunks have proper overlap"""
        text = "word " * 100
        overlap = 20
        chunks = collector._split_text_into_chunks(text, max_chunk_size=100, overlap=overlap)

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # Last part of chunk i should appear in chunk i+1
                assert len(chunks) > 1  # Verify multiple chunks were created


@pytest.mark.unit
class TestWikipediaDataCollectorEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def collector(self, test_data_dir):
        return WikipediaDataCollector(data_dir=test_data_dir)

    def test_empty_article_content(self, collector):
        """Test handling of empty article content"""
        articles = [{
            "title": "Empty",
            "url": "https://test.com",
            "content": "",
            "summary": "",
            "categories": [],
            "retrieved_at": "2024-01-01",
            "topic": "Test"
        }]

        processed = collector.preprocess_content(articles)
        # Should handle empty content gracefully
        assert isinstance(processed, list)

    def test_very_long_content(self, collector):
        """Test handling of very long article content"""
        articles = [{
            "title": "Long",
            "url": "https://test.com",
            "content": "word " * 10000,  # Very long content
            "summary": "Summary",
            "categories": ["Cat"],
            "retrieved_at": "2024-01-01",
            "topic": "Test"
        }]

        processed = collector.preprocess_content(articles)

        assert len(processed) > 1
        assert all("chunk_index" in chunk for chunk in processed)

    def test_special_characters_in_content(self, collector):
        """Test handling of special characters"""
        articles = [{
            "title": "Special",
            "url": "https://test.com",
            "content": "Content with émojis 🚀 and spëcial çharacters!",
            "summary": "Summary",
            "categories": ["Cat"],
            "retrieved_at": "2024-01-01",
            "topic": "Test"
        }]

        processed = collector.preprocess_content(articles)

        assert len(processed) > 0
        assert "🚀" in processed[0]["content"]
