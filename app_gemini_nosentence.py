"""
Wikipedia RAG System with Google Gemini - No SentenceTransformers Version
Completely bypasses sentence_transformers dependency conflicts
Uses OpenAI embeddings as fallback or simple TF-IDF for demo purposes
"""

import streamlit as st
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any, Optional
import wikipedia
import re
from collections import Counter
import math

# Try to import our core modules
try:
    import google.generativeai as genai
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError as e:
    st.error(f"Google Gemini dependencies not available: {e}")
    GOOGLE_AVAILABLE = False

# Try FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Wikipedia RAG - Gemini (No SentenceTransformers)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleEmbedder:
    """Simple embedding using TF-IDF or OpenAI as fallback"""
    
    def __init__(self, method="tfidf"):
        self.method = method
        self.vocabulary = {}
        self.idf_values = {}
        self.dimension = 384  # Default dimension
        
        if method == "openai":
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.dimension = 1536  # OpenAI embedding dimension
            except ImportError:
                st.warning("OpenAI not available, falling back to TF-IDF")
                self.method = "tfidf"
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Simple text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary for TF-IDF"""
        if self.method != "tfidf":
            return
            
        all_words = set()
        doc_freq = {}
        
        for text in texts:
            words = set(self._preprocess_text(text))
            all_words.update(words)
            
            for word in words:
                doc_freq[word] = doc_freq.get(word, 0) + 1
        
        self.vocabulary = {word: i for i, word in enumerate(sorted(all_words))}
        total_docs = len(texts)
        self.idf_values = {
            word: math.log(total_docs / freq)
            for word, freq in doc_freq.items()
        }
        self.dimension = len(self.vocabulary)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to embeddings"""
        if self.method == "openai":
            return self._openai_embeddings(texts)
        else:
            return self._tfidf_embeddings(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to embeddings (already fitted)"""
        return self.fit_transform(texts)
    
    def _openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Use OpenAI embeddings"""
        try:
            embeddings = []
            for text in texts:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embeddings.append(response.data[0].embedding)
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            st.error(f"OpenAI embedding error: {e}")
            # Fallback to TF-IDF
            self.method = "tfidf"
            return self._tfidf_embeddings(texts)
    
    def _tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Simple TF-IDF embeddings"""
        if not self.vocabulary:
            self._build_vocabulary(texts)
        
        embeddings = []
        for text in texts:
            words = self._preprocess_text(text)
            word_count = Counter(words)
            total_words = len(words)
            
            embedding = np.zeros(self.dimension)
            for word, count in word_count.items():
                if word in self.vocabulary:
                    tf = count / total_words
                    idf = self.idf_values.get(word, 1.0)
                    embedding[self.vocabulary[word]] = tf * idf
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)

class SimpleVectorStore:
    """Simple vector store without external dependencies"""
    
    def __init__(self, embedding_method="tfidf"):
        self.embedder = SimpleEmbedder(method=embedding_method)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.use_faiss = FAISS_AVAILABLE
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the store"""
        self.documents = documents
        texts = [doc['content'] for doc in documents]
        
        st.info(f"Creating embeddings using {self.embedder.method}...")
        self.embeddings = self.embedder.fit_transform(texts)
        
        if self.use_faiss and self.embeddings is not None:
            # Use FAISS for efficient search
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            # Normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            st.success(f"FAISS index built with {self.index.ntotal} vectors")
        else:
            st.info("Using simple similarity search (no FAISS)")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.embeddings is None:
            return []
        
        # Get query embedding
        query_embedding = self.embedder.transform([query])
        
        if self.use_faiss and self.index is not None:
            # FAISS search
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    result = self.documents[idx].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
            return results
        
        else:
            # Simple cosine similarity search
            query_norm = query_embedding[0]
            similarities = []
            
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = np.dot(query_norm, doc_embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            results = []
            for similarity, idx in similarities[:k]:
                result = self.documents[idx].copy()
                result['similarity_score'] = float(similarity)
                results.append(result)
            
            return results

class WikipediaDataCollector:
    """Collect and process Wikipedia articles"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
    
    def search_wikipedia(self, query: str, max_results: int = 3) -> List[str]:
        """Search Wikipedia for articles"""
        try:
            return wikipedia.search(query, results=max_results)
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def get_article_content(self, title: str) -> Optional[str]:
        """Get full article content"""
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return page.content
        except Exception as e:
            logger.error(f"Error fetching article '{title}': {e}")
            return None
    
    def collect_articles_by_topics(self, topics: List[str], articles_per_topic: int = 3) -> List[Dict[str, Any]]:
        """Collect articles by topics"""
        all_articles = []
        
        for topic in topics:
            st.info(f"Searching for articles about: {topic}")
            article_titles = self.search_wikipedia(topic, articles_per_topic)
            
            for title in article_titles[:articles_per_topic]:
                content = self.get_article_content(title)
                if content:
                    article = {
                        'title': title,
                        'content': content,
                        'topic': topic,
                        'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    }
                    all_articles.append(article)
                    st.write(f"✅ Collected: {title}")
        
        return all_articles
    
    def preprocess_content(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split articles into chunks"""
        processed_chunks = []
        
        for article in articles:
            texts = self.text_splitter.split_text(article['content'])
            
            for i, text in enumerate(texts):
                chunk = {
                    'content': text,
                    'title': article['title'],
                    'topic': article['topic'],
                    'url': article['url'],
                    'chunk_id': i
                }
                processed_chunks.append(chunk)
        
        return processed_chunks

class SimpleRAGPipeline:
    """Simple RAG pipeline using Google Gemini"""
    
    def __init__(self, api_key: str):
        if not GOOGLE_AVAILABLE:
            raise ValueError("Google Gemini dependencies not available")

        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.vector_store = None
    
    def initialize_vector_store(self, documents: List[Dict[str, Any]], embedding_method="tfidf"):
        """Initialize vector store with documents"""
        self.vector_store = SimpleVectorStore(embedding_method=embedding_method)
        self.vector_store.add_documents(documents)
    
    def generate_answer(self, query: str, max_sources: int = 3) -> Dict[str, Any]:
        """Generate answer using RAG"""
        if not self.vector_store:
            return {
                'answer': "Vector store not initialized. Please collect data first.",
                'sources': [],
                'total_tokens': 0
            }
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query, k=max_sources)
        
        if not relevant_docs:
            return {
                'answer': "No relevant information found in the knowledge base.",
                'sources': [],
                'total_tokens': 0
            }
        
        # Build context
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"Source {i}:\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context from Wikipedia articles, please answer the question. 
        Provide a comprehensive answer and cite the sources used.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            
            return {
                'answer': response.text,
                'sources': relevant_docs,
                'total_tokens': len(prompt.split()) + len(response.text.split())  # Rough estimate
            }
        
        except Exception as e:
            return {
                'answer': f"Error generating response: {e}",
                'sources': relevant_docs,
                'total_tokens': 0
            }

# Main Streamlit app
def main():
    st.title("🤖 Wikipedia RAG System with Google Gemini")
    st.markdown("**(No SentenceTransformers Version - Dependency Conflict Free)**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Setup")
        
        # API Key check
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and api_key != "your_google_api_key_here":
            st.success("✅ Google API Key configured")
        else:
            st.error("❌ Google API Key not configured")
            st.info("Please set GOOGLE_API_KEY in your .env file")
        
        # Embedding method selection
        embedding_method = st.selectbox(
            "Embedding Method:",
            ["tfidf", "openai"],
            help="TF-IDF is free but less accurate. OpenAI requires API key but more accurate."
        )
        
        st.markdown("---")
        
        # Data Collection
        st.subheader("📚 Data Collection")
        
        if st.button("🔄 Collect Sample Data"):
            if not os.path.exists("data"):
                os.makedirs("data")
            
            collector = WikipediaDataCollector()
            
            # Define topics
            topics = [
                "Artificial Intelligence",
                "Machine Learning", 
                "Deep Learning",
                "Natural Language Processing",
                "Computer Vision",
                "Neural Networks"
            ]
            
            with st.spinner("Collecting Wikipedia articles..."):
                articles = collector.collect_articles_by_topics(topics, articles_per_topic=2)
                processed_articles = collector.preprocess_content(articles)
            
            # Save articles
            with open("data/processed_wikipedia_articles.json", "w", encoding="utf-8") as f:
                json.dump(processed_articles, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ Collected {len(processed_articles)} article chunks")
            st.session_state.data_collected = True
        
        # System initialization
        st.subheader("⚙️ System Initialization")
        
        if st.button("🚀 Initialize RAG System"):
            if not os.path.exists("data/processed_wikipedia_articles.json"):
                st.error("Please collect data first!")
                return
            
            if not api_key or api_key == "your_google_api_key_here":
                st.error("Please configure Google API key first!")
                return
            
            try:
                with st.spinner("Initializing RAG system..."):
                    # Load articles
                    with open("data/processed_wikipedia_articles.json", "r", encoding="utf-8") as f:
                        articles = json.load(f)
                    
                    # Initialize RAG pipeline
                    rag_pipeline = SimpleRAGPipeline(api_key)
                    rag_pipeline.initialize_vector_store(articles, embedding_method=embedding_method)
                    
                    # Store in session state
                    st.session_state.rag_pipeline = rag_pipeline
                    st.session_state.system_initialized = True
                
                st.success("✅ RAG system initialized successfully!")
                
            except Exception as e:
                st.error(f"Initialization error: {e}")
        
        # Status
        st.markdown("---")
        st.subheader("📊 Status")
        
        if hasattr(st.session_state, 'data_collected'):
            st.write("✅ Data collected")
        else:
            st.write("⏳ Data not collected")
        
        if hasattr(st.session_state, 'system_initialized'):
            st.write("✅ System initialized")
        else:
            st.write("⏳ System not initialized")
    
    # Main content area
    if not hasattr(st.session_state, 'system_initialized'):
        st.info("👈 Please use the sidebar to collect data and initialize the system first.")
        
        # Show available fallback options
        st.markdown("## 🔧 Dependency-Free Features")
        st.markdown("""
        This version completely avoids sentence-transformers conflicts by using:
        - **TF-IDF Embeddings**: Simple but effective text vectorization
        - **OpenAI Embeddings**: High-quality embeddings (requires OpenAI API key)
        - **Pure Python Implementation**: No complex ML dependencies
        """)
        
    else:
        # Q&A Interface
        st.markdown("## 💬 Ask Questions")
        
        # Example questions
        example_questions = [
            "What is machine learning?",
            "How does deep learning work?",
            "What are neural networks?",
            "Explain artificial intelligence",
            "What is computer vision?"
        ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about AI and ML?"
            )
        
        with col2:
            selected_example = st.selectbox("Or select example:", [""] + example_questions)
            if selected_example:
                user_question = selected_example
        
        if st.button("🔍 Ask Question", disabled=not user_question):
            if user_question:
                with st.spinner("Generating answer..."):
                    result = st.session_state.rag_pipeline.generate_answer(user_question)
                
                # Display answer
                st.markdown("### 🤖 Answer")
                st.write(result['answer'])
                
                # Display sources
                if result['sources']:
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"{i}. {source['title']} (Score: {source['similarity_score']:.3f})"):
                            st.write(source['content'])
                            if 'url' in source:
                                st.markdown(f"[Wikipedia Link]({source['url']})")
                
                # Display metrics
                st.markdown("### 📊 Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sources Used", len(result['sources']))
                
                with col2:
                    st.metric("Estimated Tokens", result['total_tokens'])
                
                with col3:
                    embedding_method_display = st.session_state.rag_pipeline.vector_store.embedder.method
                    st.metric("Embedding Method", embedding_method_display.upper())
        
        # Chat History (simple version)
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if user_question and st.button("💾 Save to History"):
            st.session_state.chat_history.append({
                'question': user_question,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Question saved to history!")
        
        if st.session_state.chat_history:
            st.markdown("### 📝 Question History")
            for i, item in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                st.write(f"{i}. **{item['timestamp']}**: {item['question']}")

if __name__ == "__main__":
    main()