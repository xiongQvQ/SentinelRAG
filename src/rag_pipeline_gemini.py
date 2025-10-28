"""
RAG Pipeline Implementation using LangChain with Google Gemini
Implements the complete Retrieval-Augmented Generation workflow
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.schema import BaseRetriever, Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import get_openai_callback
from pydantic import Field
import google.generativeai as genai

try:
    from .vector_store import VectorStoreManager
except ImportError:
    from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomFAISSRetriever(BaseRetriever):
    """Custom retriever that wraps our FAISS vector store"""
    
    vector_store_manager: VectorStoreManager = Field(...)
    k: int = Field(default=5)
    
    def __init__(self, vector_store_manager: VectorStoreManager, k: int = 5, **kwargs):
        super().__init__(vector_store_manager=vector_store_manager, k=k, **kwargs)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query"""
        results = self.vector_store_manager.search_documents(query, k=self.k)
        
        documents = []
        for result in results:
            # Create LangChain Document
            doc = Document(
                page_content=result['content'],
                metadata={
                    'title': result['title'],
                    'url': result['url'],
                    'topic': result.get('topic', 'unknown'),
                    'similarity_score': result['similarity_score'],
                    'chunk_index': result.get('chunk_index', 0),
                    'source': f"{result['title']} (Score: {result['similarity_score']:.3f})"
                }
            )
            documents.append(doc)
        
        return documents
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents"""
        return self.get_relevant_documents(query)

class RAGPipeline:
    """Complete RAG pipeline with Google Gemini integration"""
    
    def __init__(self,
                 google_api_key: str,
                 vector_store_dir: str = "vector_store",
                 model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.1):
        """
        Initialize RAG pipeline with Google Gemini
        
        Args:
            google_api_key: Google API key for Gemini
            vector_store_dir: Directory containing vector store
            model_name: Google Gemini model to use
            temperature: Temperature for response generation
        """
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize components
        self.vector_store_manager = VectorStoreManager(vector_store_dir)
        self.llm = None
        self.retriever = None
        self.qa_chain = None
        self.chat_chain = None
        self.memory = None
        
        # Configure Google Gemini
        genai.configure(api_key=google_api_key)
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
    def initialize(self, articles_file: str = "data/processed_wikipedia_articles.json"):
        """Initialize all components"""
        logger.info("Initializing RAG pipeline with Google Gemini...")
        
        # Initialize vector store
        self.vector_store_manager.initialize_from_articles(articles_file)
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=self.google_api_key,
            convert_system_message_to_human=True
        )
        
        # Initialize retriever
        self.retriever = CustomFAISSRetriever(self.vector_store_manager, k=5)
        
        # Initialize memory for conversational chain
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Remember last 5 exchanges
        )
        
        # Create QA chain
        self._create_qa_chain()
        
        # Create conversational chain
        self._create_chat_chain()
        
        logger.info("RAG pipeline initialized successfully with Google Gemini!")
    
    def _create_qa_chain(self):
        """Create the Q&A chain"""
        # Custom prompt template for Gemini
        qa_template = """You are a helpful AI assistant that answers questions based on provided context from Wikipedia articles about AI, machine learning, and related topics.

Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Please provide a comprehensive and accurate answer based on the context. Include relevant details and mention the sources when possible. Be clear and informative in your response.

Answer:"""
        
        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
    
    def _create_chat_chain(self):
        """Create conversational chain"""
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question using the Q&A chain"""
        if self.qa_chain is None:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        logger.info(f"Processing question with Gemini: {question}")
        
        try:
            result = self.qa_chain({"query": question})
            
            # Format response (Gemini doesn't have token tracking like OpenAI)
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "title": doc.metadata["title"],
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "url": doc.metadata["url"],
                        "similarity_score": doc.metadata["similarity_score"]
                    }
                    for doc in result["source_documents"]
                ],
                "usage": {
                    "model": self.model_name,
                    "note": "Google Gemini API usage tracking not available"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise
    
    def chat(self, message: str) -> Dict[str, Any]:
        """Have a conversation using the chat chain"""
        if self.chat_chain is None:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        logger.info(f"Processing chat message with Gemini: {message}")
        
        try:
            result = self.chat_chain({"question": message})
            
            # Format response
            response = {
                "message": message,
                "response": result["answer"],
                "source_documents": [
                    {
                        "title": doc.metadata["title"],
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "url": doc.metadata["url"],
                        "similarity_score": doc.metadata["similarity_score"]
                    }
                    for doc in result["source_documents"]
                ],
                "usage": {
                    "model": self.model_name,
                    "note": "Google Gemini API usage tracking not available"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            raise
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> str:
        """Get relevant context for a query"""
        if self.vector_store_manager is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store_manager.get_context_for_query(query, max_context_length)
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "initialized": self.llm is not None,
            "memory_size": len(self.memory.chat_memory.messages) if self.memory else 0,
            "provider": "Google Gemini"
        }
        
        if self.vector_store_manager:
            vector_stats = self.vector_store_manager.vector_store.get_stats()
            stats.update(vector_stats)
        
        return stats

def main():
    """Example usage"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY in your .env file")
        return
    
    # Initialize pipeline
    pipeline = RAGPipeline(api_key)
    
    try:
        # Initialize with articles
        pipeline.initialize()
        
        # Example questions
        questions = [
            "What is machine learning?",
            "How does deep learning work?",
            "What are the applications of natural language processing?",
            "Explain the difference between supervised and unsupervised learning"
        ]
        
        print("RAG System Demo with Google Gemini\n" + "="*60)
        
        for question in questions:
            print(f"\nQuestion: {question}")
            print("-" * 40)
            
            response = pipeline.ask_question(question)
            print(f"Answer: {response['answer']}\n")
            
            print("Sources:")
            for i, source in enumerate(response['source_documents'][:3], 1):
                print(f"{i}. {source['title']} (Score: {source['similarity_score']:.3f})")
            
            print(f"\nModel: {response['usage']['model']}")
            print("="*60)
    
    except Exception as e:
        logger.error(f"Error running demo: {e}")

if __name__ == "__main__":
    main()