#!/usr/bin/env python3
"""
Demo Script for Wikipedia RAG System
Demonstrates the complete pipeline functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from src.data_collector import WikipediaDataCollector
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_data_collection():
    """Demonstrate data collection functionality"""
    print("\n" + "="*60)
    print("📚 DEMO: Data Collection from Wikipedia")
    print("="*60)
    
    collector = WikipediaDataCollector()
    
    # Define topics for demo
    topics = [
        "Artificial Intelligence",
        "Machine Learning", 
        "Deep Learning",
        "Natural Language Processing"
    ]
    
    print(f"Collecting articles for topics: {topics}")
    print("This may take 30-60 seconds...")
    
    try:
        # Collect articles
        articles = collector.collect_articles_by_topics(topics, articles_per_topic=2)
        print(f"✅ Collected {len(articles)} raw articles")
        
        # Preprocess
        processed_articles = collector.preprocess_content(articles)
        print(f"✅ Created {len(processed_articles)} text chunks")
        
        # Save
        collector.save_articles(processed_articles, "demo_articles.json")
        print("✅ Saved processed articles")
        
        # Show sample
        if processed_articles:
            sample = processed_articles[0]
            print(f"\nSample chunk:")
            print(f"Title: {sample['title']}")
            print(f"Content: {sample['content'][:200]}...")
            print(f"Topic: {sample['topic']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        return False

def demo_vector_store():
    """Demonstrate vector store functionality"""
    print("\n" + "="*60)
    print("🔍 DEMO: Vector Store and Similarity Search")
    print("="*60)
    
    try:
        # Initialize vector store manager
        manager = VectorStoreManager()
        
        # Load or build index
        articles_file = "data/demo_articles.json"
        if not os.path.exists(articles_file):
            print("❌ Demo articles not found. Run data collection first.")
            return False
        
        print("Building vector index...")
        manager.initialize_from_articles(articles_file, force_rebuild=True)
        
        # Test search
        test_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "What are neural networks?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = manager.search_documents(query, k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title']} (Score: {result['similarity_score']:.3f})")
                print(f"     {result['content'][:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in vector store demo: {e}")
        return False

def demo_rag_pipeline():
    """Demonstrate RAG pipeline functionality"""
    print("\n" + "="*60)
    print("🤖 DEMO: RAG Pipeline Question Answering")
    print("="*60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        return False
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(api_key)
        
        # Check if we have data
        articles_file = "data/demo_articles.json"
        if not os.path.exists(articles_file):
            print("❌ Demo articles not found. Run data collection first.")
            return False
        
        print("Initializing RAG pipeline...")
        pipeline.initialize(articles_file)
        
        # Test questions
        questions = [
            "What is machine learning and how does it work?",
            "Explain the difference between artificial intelligence and machine learning",
            "What are the main applications of deep learning?"
        ]
        
        print(f"\nTesting {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*50}")
            print(f"Question {i}: {question}")
            print("-" * 50)
            
            try:
                response = pipeline.ask_question(question)
                
                print(f"Answer: {response['answer']}\n")
                
                print("Top Sources:")
                for j, source in enumerate(response['source_documents'][:2], 1):
                    print(f"  {j}. {source['title']} (Score: {source['similarity_score']:.3f})")
                
                print(f"\nUsage: {response['usage']['total_tokens']} tokens | ${response['usage']['total_cost']:.4f}")
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
        
        # Test conversational capability
        print(f"\n{'='*50}")
        print("Testing Conversational Memory")
        print("-" * 50)
        
        conv_questions = [
            "What is deep learning?",
            "Can you give me more details about neural networks?",
            "How is this different from traditional programming?"
        ]
        
        for question in conv_questions:
            print(f"\n💬 You: {question}")
            try:
                response = pipeline.chat(question)
                print(f"🤖 Assistant: {response['response']}")
            except Exception as e:
                print(f"❌ Error in conversation: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in RAG pipeline demo: {e}")
        return False

def demo_full_system():
    """Run complete system demonstration"""
    print("🚀 Wikipedia RAG System - Complete Demo")
    print("This demo will showcase all components of the RAG system")
    print("Estimated time: 2-3 minutes")
    
    input("\nPress Enter to start the demo...")
    
    # Step 1: Data Collection
    success1 = demo_data_collection()
    if not success1:
        print("❌ Demo failed at data collection stage")
        return False
    
    input("\nPress Enter to continue to vector store demo...")
    
    # Step 2: Vector Store
    success2 = demo_vector_store()
    if not success2:
        print("❌ Demo failed at vector store stage")
        return False
    
    input("\nPress Enter to continue to RAG pipeline demo...")
    
    # Step 3: RAG Pipeline
    success3 = demo_rag_pipeline()
    if not success3:
        print("❌ Demo failed at RAG pipeline stage")
        return False
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("All components working correctly:")
    print("✅ Wikipedia data collection")
    print("✅ Vector store and similarity search")
    print("✅ RAG pipeline with Q&A capabilities")
    print("✅ Conversational memory")
    print("\nYou can now run the Streamlit app with: streamlit run app.py")
    
    return True

def quick_test():
    """Quick functionality test"""
    print("🧪 Quick System Test")
    print("="*30)
    
    # Check environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found")
        return False
    else:
        print("✅ OpenAI API key found")
    
    # Check if data exists
    if os.path.exists("data/demo_articles.json"):
        print("✅ Demo data exists")
    else:
        print("⚠️ Demo data not found")
    
    # Check if vector store exists
    if os.path.exists("vector_store/faiss_index.index"):
        print("✅ Vector store exists")
    else:
        print("⚠️ Vector store not found")
    
    print("\nSystem ready for demo!")
    return True

def main():
    """Main demo function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "data":
            demo_data_collection()
        elif sys.argv[1] == "vector":
            demo_vector_store()
        elif sys.argv[1] == "rag":
            demo_rag_pipeline()
        else:
            print("Usage: python demo.py [test|data|vector|rag|full]")
    else:
        demo_full_system()

if __name__ == "__main__":
    main()