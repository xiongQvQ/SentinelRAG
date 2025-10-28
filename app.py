"""
Streamlit Web Interface for RAG System
Interactive demo application for the Wikipedia RAG system
"""

import streamlit as st
import os
import json
from datetime import datetime
import logging
from dotenv import load_dotenv

# Import our custom modules
from src.rag_pipeline import RAGPipeline
from src.data_collector import WikipediaDataCollector
from src.vector_store import VectorStoreManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Wikipedia RAG System Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-card {
        background-color: #f9f9f9;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .stats-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False

def check_requirements():
    """Check if all requirements are met"""
    issues = []
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        issues.append("❌ OPENAI_API_KEY not found in environment variables")
    else:
        issues.append("✅ OpenAI API key found")
    
    # Check if data exists
    articles_file = "data/processed_wikipedia_articles.json"
    if not os.path.exists(articles_file):
        issues.append("❌ Processed Wikipedia articles not found")
    else:
        with open(articles_file, 'r') as f:
            articles = json.load(f)
        issues.append(f"✅ Found {len(articles)} processed Wikipedia article chunks")
    
    # Check vector store
    vector_store_dir = "vector_store"
    if not os.path.exists(os.path.join(vector_store_dir, "faiss_index.index")):
        issues.append("⚠️ Vector store not found (will be created)")
    else:
        issues.append("✅ Vector store found")
    
    return issues

def initialize_system():
    """Initialize the RAG system"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Please set your OpenAI API key in the .env file")
            return False
        
        # Initialize pipeline
        with st.spinner("Initializing RAG pipeline..."):
            pipeline = RAGPipeline(api_key)
            pipeline.initialize()
            st.session_state.pipeline = pipeline
            st.session_state.system_initialized = True
        
        st.success("✅ RAG system initialized successfully!")
        return True
    
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False

def collect_sample_data():
    """Collect sample Wikipedia data"""
    try:
        with st.spinner("Collecting Wikipedia articles..."):
            collector = WikipediaDataCollector()
            
            # Define topics
            topics = [
                "Artificial Intelligence",
                "Machine Learning", 
                "Deep Learning",
                "Natural Language Processing",
                "Computer Vision",
                "Robotics"
            ]
            
            # Collect articles
            articles = collector.collect_articles_by_topics(topics, articles_per_topic=3)
            processed_articles = collector.preprocess_content(articles)
            collector.save_articles(processed_articles, "processed_wikipedia_articles.json")
            
            st.success(f"✅ Collected {len(processed_articles)} article chunks")
            return True
    
    except Exception as e:
        st.error(f"Failed to collect data: {str(e)}")
        return False

def display_chat_interface():
    """Display the main chat interface"""
    st.markdown('<div class="main-header"><h1>🤖 Wikipedia RAG System</h1><p>Ask questions about AI, Machine Learning, and related topics!</p></div>', unsafe_allow_html=True)
    
    # Display system stats
    with st.expander("📊 System Statistics", expanded=False):
        if st.session_state.pipeline:
            stats = st.session_state.pipeline.get_stats()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", stats.get("model_name", "Unknown"))
            with col2:
                st.metric("Documents", stats.get("num_documents", 0))
            with col3:
                st.metric("Vectors", stats.get("num_vectors", 0))
            with col4:
                st.metric("Memory Size", stats.get("memory_size", 0))
    
    # Question input
    st.markdown("### 💬 Ask a Question")
    
    # Predefined example questions
    example_questions = [
        "What is machine learning?",
        "How does deep learning work?", 
        "What are the main applications of natural language processing?",
        "Explain the difference between supervised and unsupervised learning",
        "What is computer vision used for?",
        "How do neural networks learn?"
    ]
    
    # Question input methods
    input_method = st.radio("Choose input method:", ["Type your question", "Select example question"])
    
    if input_method == "Select example question":
        question = st.selectbox("Example questions:", [""] + example_questions)
    else:
        question = st.text_area("Your question:", placeholder="Type your question here...")
    
    # Ask button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary")
    
    with col2:
        clear_memory = st.button("🗑️ Clear History")
    
    if clear_memory and st.session_state.pipeline:
        st.session_state.pipeline.clear_memory()
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()
    
    # Process question
    if ask_button and question.strip():
        if not st.session_state.pipeline:
            st.error("System not initialized. Please check the setup in the sidebar.")
            return
        
        try:
            with st.spinner("Processing your question..."):
                response = st.session_state.pipeline.ask_question(question)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "response": response,
                "timestamp": datetime.now()
            })
            
            # Display response
            display_response(response)
            
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📜 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:100]}..." if len(chat['question']) > 100 else f"Q: {chat['question']}", expanded=(i == 0)):
                display_response(chat['response'])

def display_response(response):
    """Display a formatted response"""
    # Question
    st.markdown(f'<div class="question-box"><strong>❓ Question:</strong> {response["question"]}</div>', unsafe_allow_html=True)
    
    # Answer
    st.markdown(f'<div class="answer-box"><strong>🤖 Answer:</strong><br>{response["answer"]}</div>', unsafe_allow_html=True)
    
    # Sources
    st.markdown("**📚 Sources:**")
    for i, source in enumerate(response["source_documents"][:3], 1):
        st.markdown(f"""
        <div class="source-card">
            <strong>{i}. {source['title']}</strong> (Similarity: {source['similarity_score']:.3f})<br>
            <small><a href="{source['url']}" target="_blank">🔗 Wikipedia Link</a></small><br>
            <em>{source['content']}</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Usage statistics
    usage = response["usage"]
    st.markdown(f"""
    <div class="stats-box">
        <strong>📊 Usage Statistics:</strong><br>
        Tokens: {usage['total_tokens']} | Cost: ${usage['total_cost']:.4f} | Time: {response['timestamp']}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar for system setup
    with st.sidebar:
        st.header("🔧 System Setup")
        
        # Check requirements
        st.subheader("System Status")
        requirements = check_requirements()
        for req in requirements:
            st.write(req)
        
        # Data collection
        st.subheader("📚 Data Management")
        if st.button("📥 Collect Sample Data"):
            if collect_sample_data():
                st.rerun()
        
        # System initialization
        st.subheader("⚙️ System Initialization")
        if st.button("🚀 Initialize RAG System"):
            if initialize_system():
                st.rerun()
        
        # System info
        if st.session_state.system_initialized:
            st.success("✅ System Ready!")
        else:
            st.warning("⚠️ System Not Initialized")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        with st.expander("Advanced Settings"):
            st.text_input("OpenAI Model", value="gpt-3.5-turbo", disabled=True)
            st.slider("Temperature", 0.0, 1.0, 0.1, disabled=True)
            st.number_input("Max Sources", 1, 10, 5, disabled=True)
    
    # Main content area
    if st.session_state.system_initialized:
        display_chat_interface()
    else:
        st.markdown('<div class="main-header"><h1>🤖 Wikipedia RAG System</h1><p>Welcome to the RAG demonstration!</p></div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🚀 Getting Started
        
        This is a Retrieval-Augmented Generation (RAG) system demo that uses Wikipedia articles to answer questions about AI and Machine Learning topics.
        
        ### Steps to use:
        1. **Set up your environment**: Make sure you have an OpenAI API key in your `.env` file
        2. **Collect data**: Click "Collect Sample Data" in the sidebar to gather Wikipedia articles
        3. **Initialize system**: Click "Initialize RAG System" to set up the vector store and language models
        4. **Ask questions**: Once initialized, you can ask questions about AI, ML, and related topics!
        
        ### Features:
        - 🔍 **Semantic Search**: Find relevant information using vector similarity
        - 💬 **Conversational AI**: Ask follow-up questions with context memory
        - 📚 **Source Citations**: See which Wikipedia articles were used to generate answers
        - 📊 **Usage Tracking**: Monitor token usage and costs
        
        ### Sample Topics:
        - Artificial Intelligence
        - Machine Learning
        - Deep Learning
        - Natural Language Processing
        - Computer Vision
        - Robotics
        """)

if __name__ == "__main__":
    main()