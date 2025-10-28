"""
Simplified Vector Store Implementation using FAISS
A more stable version with fewer dependencies to avoid version conflicts
"""

import os
import json
import pickle
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

try:
    import faiss
except ImportError:
    print("⚠️  FAISS not installed. Install with: pip install faiss-cpu")
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  SentenceTransformer not available due to version conflicts")
    print("Using OpenAI embeddings as fallback (requires OpenAI API key)")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Fallback to OpenAI embeddings if sentence-transformers is not available
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        import openai
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbedder:
    """Simple embedding interface that can fallback to different providers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.model = None
        self.dimension = None
        self.provider = None
        
        self._initialize_embedder()
    
    def _initialize_embedder(self):
        """Initialize the best available embedding model"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.provider = "sentence-transformers"
                logger.info(f"✅ SentenceTransformer loaded with dimension: {self.dimension}")
                return
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
        
        # Fallback to OpenAI embeddings
        if OPENAI_AVAILABLE and self.api_key:
            try:
                openai.api_key = self.api_key
                # Test with a simple embedding
                response = openai.Embedding.create(
                    input="test",
                    model="text-embedding-ada-002"
                )
                self.dimension = len(response['data'][0]['embedding'])
                self.provider = "openai"
                logger.info(f"✅ OpenAI embeddings initialized with dimension: {self.dimension}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
        
        # If all else fails, use dummy embeddings for demo purposes
        logger.warning("⚠️  Using dummy embeddings for demonstration only")
        self.dimension = 384  # Default dimension
        self.provider = "dummy"
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.provider == "sentence-transformers":
            return self.model.encode(texts, show_progress_bar=show_progress_bar).astype('float32')
        
        elif self.provider == "openai":
            embeddings = []
            for text in texts:
                try:
                    response = openai.Embedding.create(
                        input=text,
                        model="text-embedding-ada-002"
                    )
                    embeddings.append(response['data'][0]['embedding'])
                except Exception as e:
                    logger.error(f"OpenAI embedding error: {e}")
                    # Use dummy embedding as fallback
                    embeddings.append(np.random.normal(0, 1, self.dimension).tolist())
            return np.array(embeddings, dtype='float32')
        
        else:  # dummy provider
            logger.warning("Using dummy embeddings - results will not be meaningful")
            return np.random.normal(0, 1, (len(texts), self.dimension)).astype('float32')

class SimpleFAISSVectorStore:
    """Simplified FAISS-based vector store"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 vector_store_dir: str = "vector_store",
                 api_key: str = None):
        """
        Initialize vector store
        
        Args:
            model_name: Embedding model name
            vector_store_dir: Directory to store vector index and metadata
            api_key: API key for fallback embedding service
        """
        self.model_name = model_name
        self.vector_store_dir = vector_store_dir
        self.embedder = None
        self.index = None
        self.documents = []
        self.dimension = None
        
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # Initialize embedder
        self.embedder = SimpleEmbedder(model_name, api_key)
        self.dimension = self.embedder.dimension
        
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from documents"""
        if not faiss:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        logger.info(f"Building FAISS index for {len(documents)} documents...")
        
        # Extract text content for embedding
        texts = [doc['content'] for doc in documents]
        
        # Create embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store documents metadata
        self.documents = documents
        
        logger.info(f"✅ Index built successfully with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        if self.index is None or not self.embedder:
            raise ValueError("Index not built or embedder not initialized")
        
        # Create query embedding
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, index_name: str = "simple_faiss_index"):
        """Save FAISS index and metadata"""
        if not faiss or self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        index_path = os.path.join(self.vector_store_dir, f"{index_name}.index")
        faiss.write_index(self.index, index_path)
        
        # Save documents metadata
        docs_path = os.path.join(self.vector_store_dir, f"{index_name}_documents.json")
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "num_documents": len(self.documents),
            "provider": self.embedder.provider if self.embedder else "unknown"
        }
        config_path = os.path.join(self.vector_store_dir, f"{index_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Index saved to {index_path}")
        return index_path
    
    def load_index(self, index_name: str = "simple_faiss_index"):
        """Load FAISS index and metadata"""
        if not faiss:
            raise ImportError("FAISS not available")
        
        # Load configuration
        config_path = os.path.join(self.vector_store_dir, f"{index_name}_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.model_name = config["model_name"]
        self.dimension = config["dimension"]
        
        # Load FAISS index
        index_path = os.path.join(self.vector_store_dir, f"{index_name}.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load documents
        docs_path = os.path.join(self.vector_store_dir, f"{index_name}_documents.json")
        with open(docs_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        logger.info(f"✅ Index loaded with {self.index.ntotal} vectors")
        return True

class SimpleVectorStoreManager:
    """High-level manager for simple vector store operations"""
    
    def __init__(self, vector_store_dir: str = "vector_store", api_key: str = None):
        self.vector_store = SimpleFAISSVectorStore(vector_store_dir=vector_store_dir, api_key=api_key)
        self.vector_store_dir = vector_store_dir
    
    def initialize_from_articles(self, articles_file: str, force_rebuild: bool = False):
        """Initialize vector store from articles file"""
        index_exists = os.path.exists(os.path.join(self.vector_store_dir, "simple_faiss_index.index"))
        
        if index_exists and not force_rebuild:
            logger.info("Loading existing vector store...")
            self.vector_store.load_index()
        else:
            logger.info("Building new vector store...")
            
            # Load articles
            if not os.path.exists(articles_file):
                raise FileNotFoundError(f"Articles file not found: {articles_file}")
            
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Build index
            self.vector_store.build_index(articles)
            
            # Save index
            self.vector_store.save_index()
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        results = self.vector_store.search(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            result = doc.copy()
            result['similarity_score'] = score
            formatted_results.append(result)
        
        return formatted_results

def main():
    """Example usage"""
    # Initialize simple vector store manager
    manager = SimpleVectorStoreManager()
    
    # Check if we have processed articles
    articles_file = "data/processed_wikipedia_articles.json"
    if not os.path.exists(articles_file):
        print("⚠️  Please run data_collector.py first to collect Wikipedia articles")
        print("Or run: python demo_gemini.py data")
        return
    
    try:
        # Initialize vector store
        manager.initialize_from_articles(articles_file)
        
        # Test search
        query = "What is machine learning?"
        results = manager.search_documents(query, k=3)
        
        print(f"\nSearch results for: '{query}'\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (Score: {result['similarity_score']:.3f})")
            print(f"   {result['content'][:200]}...\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTry running the dependency fix script:")
        print("python fix_dependencies.py")

if __name__ == "__main__":
    main()