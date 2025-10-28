"""
Streamlit Web Interface for RAG System with Google Gemini
Interactive demo application for the Wikipedia RAG system
"""

import os
# Set environment variables BEFORE any other imports to prevent multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import json
from datetime import datetime
import logging
from dotenv import load_dotenv

# Import our custom modules
from src.rag_pipeline_gemini import RAGPipeline
from src.data_collector import WikipediaDataCollector
from src.vector_store import VectorStoreManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Wikipedia RAG System with Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4285f4 0%, #34a853 50%, #fbbc05 75%, #ea4335 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4285f4;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #34a853;
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
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .gemini-badge {
        background: linear-gradient(90deg, #4285f4, #34a853);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
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
    
    # Check Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        issues.append("❌ GOOGLE_API_KEY not found in environment variables")
    else:
        issues.append("✅ Google API key found")
    
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
    """Initialize the RAG system with Gemini"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Please set your Google API key in the .env file")
            return False
        
        # Initialize pipeline
        with st.spinner("Initializing RAG pipeline with Google Gemini..."):
            pipeline = RAGPipeline(api_key)
            pipeline.initialize()
            st.session_state.pipeline = pipeline
            st.session_state.system_initialized = True
        
        st.success("✅ RAG system initialized successfully with Google Gemini!")
        return True
    
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False

def collect_sample_data():
    """Collect sample Wikipedia data (only if no local data exists)"""
    try:
        # Check if we already have local data
        articles_file = "data/processed_wikipedia_articles.json"
        if os.path.exists(articles_file):
            with open(articles_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            st.info(f"✅ 发现本地数据: {len(existing_data)} 个文档块")
            st.info("💡 提示: 如需更新数据，请使用预下载脚本: `python predownload_data.py`")
            return True
        
        with st.spinner("Collecting Wikipedia articles..."):
            collector = WikipediaDataCollector()
            
            # Define topics (reduced for faster loading)
            topics = [
                "Artificial Intelligence",
                "Machine Learning", 
                "Deep Learning"
            ]
            
            st.warning("⚠️ 正在在线下载数据，这可能需要较长时间...")
            st.info("💡 建议使用预下载脚本: `python predownload_data.py` 来提前下载更多数据")
            
            # Collect articles (reduced count for faster loading)
            articles = collector.collect_articles_by_topics(topics, articles_per_topic=2)
            processed_articles = collector.preprocess_content(articles)
            collector.save_articles(processed_articles, "processed_wikipedia_articles.json")
            
            st.success(f"✅ Collected {len(processed_articles)} article chunks")
            st.info("🚀 为获得更好的体验，建议运行预下载脚本获取更多数据")
            return True
    
    except Exception as e:
        st.error(f"Failed to collect data: {str(e)}")
        st.error("💡 尝试运行预下载脚本: `python predownload_data.py`")
        return False

def display_chat_interface():
    """Display the main chat interface"""
    st.markdown('<div class="main-header"><h1>🤖 Wikipedia RAG System</h1><p>Powered by <span class="gemini-badge">Google Gemini</span> - Ask questions about AI, Machine Learning, and related topics!</p></div>', unsafe_allow_html=True)
    
    # Display system stats
    with st.expander("📊 System Statistics", expanded=False):
        if st.session_state.pipeline:
            stats = st.session_state.pipeline.get_stats()
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Provider", stats.get("provider", "Unknown"))
            with col2:
                st.metric("Model", stats.get("model_name", "Unknown"))
            with col3:
                st.metric("Documents", stats.get("num_documents", 0))
            with col4:
                st.metric("Vectors", stats.get("num_vectors", 0))
            with col5:
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
        "How do neural networks learn?",
        "What are the ethical considerations in AI development?",
        "How does reinforcement learning work?"
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
            with st.spinner("Processing your question with Gemini..."):
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
    
    # Answer with Gemini badge
    st.markdown(f'<div class="answer-box"><strong>🤖 Answer</strong> <span class="gemini-badge">Gemini</span><br>{response["answer"]}</div>', unsafe_allow_html=True)
    
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
        Model: {usage['model']} | {usage['note']} | Time: {response['timestamp']}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar for system setup
    with st.sidebar:
        st.header("🔧 System Setup")
        
        # Display Gemini info
        st.markdown("### 🚀 Powered by Google Gemini")
        st.info("This version uses Google's Gemini AI for enhanced performance and multimodal capabilities.")
        
        # Check requirements
        st.subheader("System Status")
        requirements = check_requirements()
        for req in requirements:
            st.write(req)
        
        # Data collection
        st.subheader("📚 Data Management")
        
        # Check if predownload script exists
        predownload_script = "predownload_data.py"
        if os.path.exists(predownload_script):
            st.info("💡 建议使用预下载脚本获取更多数据:")
            st.code("python predownload_data.py", language="bash")
        
        if st.button("📥 收集数据 (在线)"):
            if collect_sample_data():
                st.rerun()
        
        # Display local data status
        articles_file = "data/processed_wikipedia_articles.json"
        if os.path.exists(articles_file):
            try:
                with open(articles_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                st.success(f"📊 本地数据: {len(existing_data)} 个文档块")
                
                # Show data stats if available
                stats_file = "data/data_stats.json"
                if os.path.exists(stats_file):
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                    with st.expander("📈 数据详情"):
                        st.write(f"独特文章: {stats.get('unique_articles', 'N/A')}")
                        st.write(f"话题数量: {len(stats.get('topics', {}))}")
                        topics = list(stats.get('topics', {}).keys())[:5]
                        if topics:
                            st.write(f"主要话题: {', '.join(topics)}")
            except:
                st.warning("⚠️ 本地数据文件可能损坏")
        else:
            st.warning("❌ 没有本地数据")
        
        # System initialization
        st.subheader("⚙️ System Initialization")
        if st.button("🚀 Initialize RAG System"):
            if initialize_system():
                st.rerun()
        
        # System info
        if st.session_state.system_initialized:
            st.success("✅ System Ready with Gemini!")
        else:
            st.warning("⚠️ System Not Initialized")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        with st.expander("Advanced Settings"):
            st.text_input("Gemini Model", value="gemini-1.5-flash", disabled=True)
            st.slider("Temperature", 0.0, 1.0, 0.1, disabled=True)
            st.number_input("Max Sources", 1, 10, 5, disabled=True)
            
        # Model comparison
        st.subheader("🆚 Why Gemini?")
        st.markdown("""
        - **Faster responses** than OpenAI
        - **Lower cost** for many use cases
        - **Multimodal capabilities** (text, images)
        - **Large context window** (1M+ tokens)
        - **Latest AI technology** from Google
        """)
    
    # Main content area
    if st.session_state.system_initialized:
        display_chat_interface()
    else:
        st.markdown('<div class="main-header"><h1>🤖 Wikipedia RAG System</h1><p>Powered by Google Gemini AI!</p></div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🚀 Getting Started
        
        This is a Retrieval-Augmented Generation (RAG) system demo that uses Wikipedia articles to answer questions about AI and Machine Learning topics, powered by **Google Gemini**.
        
        ### Why Google Gemini?
        - 🚀 **Faster**: Optimized for speed and efficiency
        - 💰 **Cost-effective**: Often lower cost than alternatives
        - 🧠 **Advanced**: Latest AI technology from Google
        - 📚 **Large context**: Can handle extensive context windows
        - 🔄 **Multimodal**: Supports text and images
        
        ### 快速开始步骤:
        1. **环境设置**: 确保在 `.env` 文件中有 Google API 密钥
        2. **预下载数据** (推荐): 运行 `python predownload_data.py` 提前下载所有数据
        3. **初始化系统**: 点击侧边栏的 "Initialize RAG System"
        4. **开始提问**: 系统初始化完成后就可以提问了！
        
        ### 🎯 推荐流程:
        ```bash
        # 1. 预下载数据 (避免启动时长时间等待)
        python predownload_data.py
        
        # 2. 启动应用
        streamlit run app_gemini.py
        ```
        
        ### Features:
        - 🔍 **Semantic Search**: Find relevant information using vector similarity
        - 💬 **Conversational AI**: Ask follow-up questions with context memory
        - 📚 **Source Citations**: See which Wikipedia articles were used to generate answers
        - 🤖 **Gemini Power**: Enhanced responses with Google's latest AI
        
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