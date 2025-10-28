"""
Wikipedia Data Collector for RAG System
Collects and preprocesses Wikipedia articles for vector database
"""

import wikipedia
import os
import json
from typing import List, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaDataCollector:
    """Collects Wikipedia articles for RAG knowledge base"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def search_articles(self, query: str, num_articles: int = 10) -> List[str]:
        """Search for Wikipedia articles by query"""
        try:
            search_results = wikipedia.search(query, results=num_articles)
            logger.info(f"Found {len(search_results)} articles for query: {query}")
            return search_results
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            return []
    
    def get_article_content(self, title: str) -> Dict[str, Any]:
        """Get full content of a Wikipedia article"""
        try:
            page = wikipedia.page(title)
            return {
                "title": page.title,
                "url": page.url,
                "content": page.content,
                "summary": page.summary,
                "categories": page.categories[:10],  # Limit categories
                "retrieved_at": datetime.now().isoformat()
            }
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation by taking the first option
            logger.warning(f"Disambiguation for {title}, using: {e.options[0]}")
            return self.get_article_content(e.options[0])
        except wikipedia.exceptions.PageError:
            logger.error(f"Page not found: {title}")
            return None
        except Exception as e:
            logger.error(f"Error getting article {title}: {e}")
            return None
    
    def collect_articles_by_topics(self, topics: List[str], articles_per_topic: int = 5) -> List[Dict[str, Any]]:
        """Collect articles for multiple topics"""
        all_articles = []
        
        for topic in topics:
            logger.info(f"Collecting articles for topic: {topic}")
            article_titles = self.search_articles(topic, articles_per_topic)
            
            for title in article_titles:
                article = self.get_article_content(title)
                if article:
                    article["topic"] = topic
                    all_articles.append(article)
        
        return all_articles
    
    def preprocess_content(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess articles for RAG system"""
        processed_articles = []
        
        for article in articles:
            # Split content into chunks for better retrieval
            content = article["content"]
            chunks = self._split_text_into_chunks(content, max_chunk_size=1000)
            
            for i, chunk in enumerate(chunks):
                processed_chunk = {
                    "id": f"{article['title']}_chunk_{i}",
                    "title": article["title"],
                    "url": article["url"],
                    "content": chunk,
                    "summary": article["summary"],
                    "topic": article.get("topic", "unknown"),
                    "categories": article["categories"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "retrieved_at": article["retrieved_at"]
                }
                processed_articles.append(processed_chunk)
        
        return processed_articles
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the break point
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + max_chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = max(start + 1, end - overlap)
            
            if start >= len(text):
                break
        
        return chunks
    
    def save_articles(self, articles: List[Dict[str, Any]], filename: str = "wikipedia_articles.json"):
        """Save articles to JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath
    
    def load_articles(self, filename: str = "wikipedia_articles.json") -> List[Dict[str, Any]]:
        """Load articles from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {filepath}")
        return articles

def main():
    """Example usage"""
    collector = WikipediaDataCollector()
    
    # Define topics for collecting articles
    topics = [
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Robotics"
    ]
    
    # Collect articles
    logger.info("Starting article collection...")
    articles = collector.collect_articles_by_topics(topics, articles_per_topic=3)
    logger.info(f"Collected {len(articles)} articles")
    
    # Preprocess for RAG
    processed_articles = collector.preprocess_content(articles)
    logger.info(f"Created {len(processed_articles)} text chunks")
    
    # Save to file
    collector.save_articles(processed_articles, "processed_wikipedia_articles.json")
    logger.info("Data collection completed!")

if __name__ == "__main__":
    main()