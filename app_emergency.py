"""
Emergency Streamlit Web Interface for RAG System
Uses simplified dependencies to avoid version conflicts
"""

import streamlit as st
import os
import json
from datetime import datetime
import logging
from dotenv import load_dotenv

# Try to import our modules with fallbacks
try:
    from src.data_collector import WikipediaDataCollector
    from src.vector_store_simple import SimpleVectorStoreManager
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please run: python fix_dependencies.py")
    COMPONENTS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="RAG System Emergency Mode",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .emergency-notice {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def show_emergency_notice():
    """Show emergency mode notice"""
    st.markdown("""
    <div class="emergency-notice">
        <h3>🚨 Emergency Mode Active</h3>
        <p>This version uses simplified dependencies to avoid common version conflicts.</p>
        <p>If you see import errors, please run: <code>python fix_dependencies.py</code></p>
    </div>
    """, unsafe_allow_html=True)

def check_environment():
    """Check environment and dependencies"""
    issues = []
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        issues.append("❌ GOOGLE_API_KEY not found")
    else:
        issues.append("✅ Google API key found")
    
    # Check components
    if COMPONENTS_AVAILABLE:
        issues.append("✅ Core components loaded")
    else:
        issues.append("❌ Core components failed to load")
    
    # Check data
    if os.path.exists("data/processed_wikipedia_articles.json"):
        issues.append("✅ Sample data available")
    else:
        issues.append("❌ No sample data found")
    
    return issues

def collect_sample_data():
    """Collect sample data with error handling"""
    if not COMPONENTS_AVAILABLE:
        st.error("Components not available. Please fix dependencies first.")
        return False
    
    try:
        with st.spinner("Collecting sample data..."):
            collector = WikipediaDataCollector()
            
            topics = [
                "Artificial Intelligence",
                "Machine Learning",
                "Deep Learning"
            ]
            
            articles = collector.collect_articles_by_topics(topics, articles_per_topic=2)
            processed_articles = collector.preprocess_content(articles)
            collector.save_articles(processed_articles, "processed_wikipedia_articles.json")
            
            st.success(f"✅ Collected {len(processed_articles)} article chunks")
            return True
    
    except Exception as e:
        st.error(f"Error collecting data: {e}")
        return False

def simple_qa_demo():
    """Simple Q&A demo without full RAG pipeline"""
    st.markdown("### 🔍 Simple Q&A Demo")
    
    if not COMPONENTS_AVAILABLE:
        st.warning("Full RAG functionality not available. Please fix dependencies.")
        return
    
    # Check if data exists
    articles_file = "data/processed_wikipedia_articles.json"
    if not os.path.exists(articles_file):
        st.warning("No data available. Please collect sample data first.")
        return
    
    try:
        # Initialize simple vector store
        api_key = os.getenv("GOOGLE_API_KEY")
        manager = SimpleVectorStoreManager(api_key=api_key)
        
        with st.spinner("Initializing vector store..."):
            manager.initialize_from_articles(articles_file)
        
        st.success("✅ Vector store initialized")
        
        # Simple search interface
        query = st.text_input("Enter your question:", placeholder="What is machine learning?")
        
        if st.button("🔍 Search") and query:
            try:
                with st.spinner("Searching..."):
                    results = manager.search_documents(query, k=3)
                
                st.write("**Search Results:**")
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result['title']} (Score: {result['similarity_score']:.3f})"):
                        st.write(result['content'])
                        if 'url' in result:
                            st.write(f"[Wikipedia Link]({result['url']})")
            
            except Exception as e:
                st.error(f"Search error: {e}")
    
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.info("Try using the dependency fix script: `python fix_dependencies.py`")

def show_fix_instructions():
    """Show detailed fix instructions"""
    st.markdown("### 🔧 How to Fix Dependencies")
    
    st.code("""
# Option 1: Run the fix script (recommended)
python fix_dependencies.py

# Option 2: Manual fix
pip uninstall sentence-transformers huggingface-hub transformers
pip install huggingface-hub==0.16.4 transformers==4.30.2 sentence-transformers==2.6.1

# Option 3: Use minimal requirements
pip install -r requirements-minimal.txt

# Option 4: Clean environment (most reliable)
python -m venv venv_clean
source venv_clean/bin/activate  # On Windows: venv_clean\\Scripts\\activate
pip install -r requirements-minimal.txt
    """)
    
    st.markdown("""
    **Common Issues:**
    
    1. **HuggingFace Hub conflicts**: Version mismatch between packages
    2. **LangChain version issues**: Rapid development causes compatibility issues
    3. **Torch/Transformers conflicts**: Heavy dependencies with version constraints
    
    **Recommended Solution**: Use the fix script or create a clean environment.
    """)

def main():
    """Main application"""
    st.markdown('<div class="main-header"><h1>🚨 RAG System - Emergency Mode</h1><p>Simplified version to handle dependency conflicts</p></div>', unsafe_allow_html=True)
    
    # Show emergency notice
    show_emergency_notice()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check environment
        issues = check_environment()
        for issue in issues:
            st.write(issue)
        
        st.markdown("---")
        
        # Data collection
        st.subheader("📚 Data Collection")
        if st.button("📥 Collect Sample Data"):
            collect_sample_data()
        
        st.markdown("---")
        
        # Fix instructions
        if st.button("🔧 Show Fix Instructions"):
            st.session_state.show_fix_instructions = True
    
    # Main content
    if not COMPONENTS_AVAILABLE:
        st.error("⚠️ Core components not available")
        show_fix_instructions()
    else:
        simple_qa_demo()
    
    # Show fix instructions if requested
    if hasattr(st.session_state, 'show_fix_instructions') and st.session_state.show_fix_instructions:
        show_fix_instructions()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="success-box">
        <strong>💡 Next Steps:</strong>
        <ol>
            <li>Fix dependencies using the provided instructions</li>
            <li>Use the full-featured app: <code>streamlit run app_gemini.py</code></li>
            <li>Or run the complete demo: <code>python demo_gemini.py</code></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()