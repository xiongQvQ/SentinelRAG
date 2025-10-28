"""
Secure Streamlit Web Interface for RAG System with Guardrails AI
Enhanced version with runtime safety and output validation
"""

import streamlit as st
import os
import json
from datetime import datetime
import logging
from dotenv import load_dotenv

# Import custom modules
try:
    from src.rag_pipeline_with_guardrails import SecureRAGPipeline
    from src.data_collector import WikipediaDataCollector
    from src.vector_store import VectorStoreManager
    from src.security.rate_limiter import RateLimitExceeded
    HAS_SECURE_PIPELINE = True
except ImportError:
    from src.rag_pipeline_gemini import RAGPipeline
    HAS_SECURE_PIPELINE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Secure Wikipedia RAG with Gemini & Guardrails",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with security theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4285f4 0%, #34a853 50%, #fbbc05 75%, #ea4335 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .security-badge {
        background: linear-gradient(90deg, #00C853, #00E676);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .warning-badge {
        background: #FF6F00;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
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
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{datetime.now().timestamp()}"


def check_requirements():
    """Check if all requirements are met"""
    issues = []

    # Check Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        issues.append("❌ GOOGLE_API_KEY not found in environment variables")
    else:
        issues.append("✅ Google API key found")

    # Check for processed data
    if os.path.exists("data/processed_wikipedia_articles.json"):
        issues.append("✅ Processed Wikipedia articles found")
    else:
        issues.append("⚠️  No processed articles found - run data collection first")

    # Check for vector store
    if os.path.exists("vector_store/index.faiss"):
        issues.append("✅ Vector store index found")
    else:
        issues.append("⚠️  No vector store found - will be created on initialization")

    # Check Guardrails availability
    if HAS_SECURE_PIPELINE:
        issues.append("✅ Guardrails AI integration available")
    else:
        issues.append("⚠️  Guardrails AI not available (optional)")

    return issues


def initialize_rag_system(enable_guardrails=False, enable_rate_limiting=True, enable_audit=True):
    """
    Initialize the RAG system

    注意: Guardrails默认禁用，因为需要额外的OpenAI API key
    如果需要启用Guardrails，请确保设置了OPENAI_API_KEY环境变量
    """
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("❌ Google API key not found. Please set GOOGLE_API_KEY environment variable.")
            return False

        with st.spinner("🔧 Initializing RAG system with security features..."):
            # Use Secure Pipeline if available
            if HAS_SECURE_PIPELINE:
                st.session_state.pipeline = SecureRAGPipeline(
                    google_api_key=google_api_key,
                    model_name="gemini-2.5-flash",  # 使用最新模型
                    temperature=0.1,
                    enable_guardrails=enable_guardrails,  # 默认False
                    enable_rate_limiting=enable_rate_limiting,
                    enable_audit_logging=enable_audit
                )
            else:
                # Fallback to regular pipeline
                from src.rag_pipeline_gemini import RAGPipeline
                st.session_state.pipeline = RAGPipeline(
                    google_api_key=google_api_key,
                    model_name="gemini-2.5-flash",  # 使用最新模型
                    temperature=0.1
                )

            # Initialize with data
            st.session_state.pipeline.initialize()
            st.session_state.system_initialized = True

            return True

    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        logger.error(f"Initialization error: {e}")
        return False


def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Secure Wikipedia RAG System</h1>
        <p>Powered by Google Gemini & Guardrails AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Display security features
    if HAS_SECURE_PIPELINE:
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <span class="security-badge">🔒 Input Validation</span>
            <span class="security-badge">🛡️ Output Filtering</span>
            <span class="security-badge">⏱️ Rate Limiting</span>
            <span class="security-badge">📊 Audit Logging</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <span class="warning-badge">⚠️ Running in basic mode (Guardrails not available)</span>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    initialize_session_state()
    display_header()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # System status
        st.subheader("📊 System Status")
        requirements = check_requirements()
        for req in requirements:
            st.text(req)

        st.divider()

        # Security settings (only if Guardrails available)
        if HAS_SECURE_PIPELINE:
            st.subheader("🔒 Security Settings")
            enable_guardrails = st.checkbox("Enable Guardrails AI", value=True,
                                           help="Input validation and output filtering")
            enable_rate_limiting = st.checkbox("Enable Rate Limiting", value=True,
                                              help="Prevent abuse with rate limits")
            enable_audit = st.checkbox("Enable Audit Logging", value=True,
                                      help="Log all queries for compliance")
        else:
            enable_guardrails = False
            enable_rate_limiting = False
            enable_audit = False
            st.info("💡 Install guardrails-ai for security features:\n```\npip install guardrails-ai\n```")

        st.divider()

        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            if initialize_rag_system(enable_guardrails, enable_rate_limiting, enable_audit):
                st.success("✅ System initialized successfully!")

        # Clear history button
        if st.session_state.system_initialized:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                if st.session_state.pipeline:
                    st.session_state.pipeline.clear_memory()
                st.success("✅ Chat history cleared")

        st.divider()

        # Statistics
        if st.session_state.system_initialized and st.session_state.pipeline:
            st.subheader("📊 Statistics")
            if hasattr(st.session_state.pipeline, 'get_stats'):
                stats = st.session_state.pipeline.get_stats()
                if 'pipeline' in stats:
                    st.metric("Total Queries", stats['pipeline']['total_queries'])
                    st.metric("Successful", stats['pipeline']['successful_queries'])
                    st.metric("Failed", stats['pipeline']['failed_queries'])

                    if 'guardrails' in stats:
                        st.metric("Validation Failures", stats['pipeline']['validation_failures'])

    # Main content area
    if not st.session_state.system_initialized:
        st.info("👈 Click 'Initialize System' in the sidebar to start")

        # Quick start guide
        st.subheader("📖 Quick Start Guide")
        st.markdown("""
        1. Ensure `GOOGLE_API_KEY` is set in your `.env` file
        2. Run data collection if needed: `python src/data_collector.py`
        3. Click **Initialize System** in the sidebar
        4. Start asking questions about AI and Machine Learning!

        **Security Features:**
        - **Input Validation**: Detects toxic language and unusual prompts
        - **Output Filtering**: Removes PII and sensitive information
        - **Rate Limiting**: Prevents abuse (10/min, 100/hour)
        - **Audit Logging**: Tracks all queries for compliance
        """)

    else:
        # Query interface
        st.subheader("💬 Ask Questions")

        # Input method selection
        query_mode = st.radio("Select mode:", ["Single Question", "Chat Mode"], horizontal=True)

        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What is machine learning?",
            key="question_input"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary")
        with col2:
            if HAS_SECURE_PIPELINE:
                st.caption(f"🔒 User ID: {st.session_state.user_id}")

        # Process question
        if ask_button and question:
            try:
                with st.spinner("🤔 Thinking..."):
                    # Call appropriate method based on mode
                    if query_mode == "Chat Mode":
                        if hasattr(st.session_state.pipeline, 'chat'):
                            response = st.session_state.pipeline.chat(
                                question,
                                user_id=st.session_state.user_id,
                                ip_address="streamlit_app"
                            )
                            answer = response.get('response', response.get('answer', 'No answer'))
                        else:
                            response = st.session_state.pipeline.chat(question)
                            answer = response.get('response', response.get('answer', 'No answer'))
                    else:
                        if hasattr(st.session_state.pipeline, 'ask_question'):
                            response = st.session_state.pipeline.ask_question(
                                question,
                                user_id=st.session_state.user_id,
                                ip_address="streamlit_app"
                            )
                            answer = response.get('answer', 'No answer')
                        else:
                            response = st.session_state.pipeline.ask_question(question)
                            answer = response.get('answer', 'No answer')

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.now().isoformat(),
                        "security": response.get('security', {})
                    })

                # Display answer
                st.markdown(f"""
                <div class="question-box">
                    <strong>❓ Question:</strong><br/>
                    {question}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="answer-box">
                    <strong>💡 Answer:</strong><br/>
                    {answer}
                </div>
                """, unsafe_allow_html=True)

                # Show security info if available
                if 'security' in response:
                    security_info = response['security']
                    if security_info.get('guardrails_enabled'):
                        st.success(f"✅ Input Validated | ✅ Output Validated")

                # Show sources
                if 'source_documents' in response:
                    with st.expander(f"📚 Sources ({len(response['source_documents'])} documents)"):
                        for i, doc in enumerate(response['source_documents'], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{i}. {doc['title']}</strong><br/>
                                <small>Score: {doc['similarity_score']:.3f}</small><br/>
                                <a href="{doc['url']}" target="_blank">View on Wikipedia</a>
                            </div>
                            """, unsafe_allow_html=True)

            except RateLimitExceeded:
                st.error("🚫 Rate limit exceeded! Please wait before making more requests.")

            except ValueError as e:
                if "validation failed" in str(e).lower():
                    st.error(f"🚫 Input validation failed: {str(e)}")
                else:
                    st.error(f"❌ Error: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                logger.error(f"Query error: {e}")

        # Chat history
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📜 Chat History")

            for i, entry in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {entry['question'][:50]}... ({entry['timestamp'][:19]})"):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f"**Answer:** {entry['answer']}")
                    if entry.get('security', {}).get('guardrails_enabled'):
                        st.caption("🔒 Protected by Guardrails AI")


if __name__ == "__main__":
    main()
