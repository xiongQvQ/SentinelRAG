"""
Vector Store Implementation using FAISS
Handles document embeddings and similarity search for RAG system
"""

import os
# Force single-threaded execution to prevent segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging

# Set torch to single-threaded mode
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 vector_store_dir: str = "vector_store"):
        """
        Initialize FAISS vector store
        
        Args:
            model_name: SentenceTransformer model for embeddings
            vector_store_dir: Directory to store vector index and metadata
        """
        self.model_name = model_name
        self.vector_store_dir = vector_store_dir
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.dimension = None
        
        os.makedirs(vector_store_dir, exist_ok=True)
        
    def load_embedding_model(self):
        """Load the sentence transformer model"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Disable tokenizers parallelism to prevent multiprocessing issues
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.embedding_model = SentenceTransformer(self.model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.dimension}")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        if self.embedding_model is None:
            self.load_embedding_model()

        logger.info(f"Creating embeddings for {len(texts)} texts...")
        # Disable ALL forms of parallelism to prevent segmentation fault
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            batch_size=1,              # Process one at a time
            convert_to_numpy=True,
            normalize_embeddings=False,
            device='cpu'               # Force CPU (no num_workers in 2.3.1)
        )
        return embeddings.astype('float32')
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from documents"""
        logger.info(f"Building FAISS index for {len(documents)} documents...")
        
        # Extract text content for embedding
        texts = [doc['content'] for doc in documents]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Initialize FAISS index
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store documents metadata
        self.documents = documents
        
        logger.info(f"Index built successfully with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        if self.index is None or self.embedding_model is None:
            raise ValueError("Index not built or model not loaded")
        
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, index_name: str = "faiss_index"):
        """Save FAISS index and metadata"""
        if self.index is None:
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
            "num_documents": len(self.documents)
        }
        config_path = os.path.join(self.vector_store_dir, f"{index_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Index saved to {index_path}")
        return index_path
    
    def load_index(self, index_name: str = "faiss_index"):
        """Load FAISS index and metadata"""
        # Load configuration
        config_path = os.path.join(self.vector_store_dir, f"{index_name}_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.model_name = config["model_name"]
        self.dimension = config["dimension"]
        
        # Load embedding model
        self.load_embedding_model()
        
        # Load FAISS index
        index_path = os.path.join(self.vector_store_dir, f"{index_name}.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load documents
        docs_path = os.path.join(self.vector_store_dir, f"{index_name}_documents.json")
        with open(docs_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        logger.info(f"Index loaded with {self.index.ntotal} vectors")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "num_vectors": self.index.ntotal,
            "num_documents": len(self.documents)
        }
    
    def add_documents(self, new_documents: List[Dict[str, Any]]):
        """Add new documents to existing index"""
        if self.index is None:
            # If no index exists, build from scratch
            self.build_index(new_documents)
            return
        
        logger.info(f"Adding {len(new_documents)} new documents to existing index...")
        
        # Create embeddings for new documents
        texts = [doc['content'] for doc in new_documents]
        embeddings = self.create_embeddings(texts)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Add to documents list
        self.documents.extend(new_documents)
        
        logger.info(f"Added {len(new_documents)} documents. Total: {self.index.ntotal}")


class VectorStoreManager:
    """High-level manager for vector store operations"""
    
    def __init__(self, vector_store_dir: str = "vector_store"):
        self.vector_store = FAISSVectorStore(vector_store_dir=vector_store_dir)
        self.vector_store_dir = vector_store_dir
    
    def initialize_from_articles(self, articles_file: str, force_rebuild: bool = False):
        """Initialize vector store from articles file"""
        index_exists = os.path.exists(os.path.join(self.vector_store_dir, "faiss_index.index"))
        
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
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> str:
        """Get context string for RAG pipeline"""
        results = self.search_documents(query, k=10)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result['content']
            title = result['title']
            
            # Add source information
            source_info = f"\n[Source: {title}]\n"
            part = source_info + content
            
            if current_length + len(part) > max_context_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "\n".join(context_parts)

def main():
    """Example usage"""
    # Initialize vector store manager
    manager = VectorStoreManager()
    
    # Check if we have processed articles
    articles_file = "data/processed_wikipedia_articles.json"
    if not os.path.exists(articles_file):
        print("Please run data_collector.py first to collect Wikipedia articles")
        return
    
    # Initialize vector store
    manager.initialize_from_articles(articles_file)
    
    # Test search
    query = "What is machine learning?"
    results = manager.search_documents(query, k=3)
    
    print(f"\nSearch results for: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['similarity_score']:.3f})")
        print(f"   {result['content'][:200]}...\n")

if __name__ == "__main__":
    main()