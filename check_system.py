#!/usr/bin/env python3
"""
System Check Script for Wikipedia RAG System
Quick diagnostic to identify and suggest fixes for common issues
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Python Version Check")
    print(f"   Current: {sys.version}")
    
    if sys.version_info >= (3, 8):
        print("   ✅ Compatible")
        return True
    else:
        print("   ❌ Requires Python 3.8+")
        return False

def check_basic_imports():
    """Check if basic Python packages can be imported"""
    print("\n📦 Basic Package Check")
    
    basic_packages = [
        "json", "os", "sys", "logging", "datetime", "pathlib"
    ]
    
    all_ok = True
    for package in basic_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            all_ok = False
    
    return all_ok

def check_key_dependencies():
    """Check critical dependencies with detailed error reporting"""
    print("\n🔑 Key Dependencies Check")
    
    dependencies = [
        ("streamlit", "Web interface"),
        ("dotenv", "Environment variables (python-dotenv)"),
        ("wikipedia", "Wikipedia API access"),
        ("numpy", "Numerical computing"),
        ("pandas", "Data manipulation")
    ]
    
    results = {}
    
    for package, description in dependencies:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {package} ({version}) - {description}")
            results[package] = True
        except ImportError as e:
            print(f"   ❌ {package} - {description}")
            print(f"      Error: {e}")
            results[package] = False
    
    return results

def check_ai_dependencies():
    """Check AI/ML dependencies that commonly have conflicts"""
    print("\n🤖 AI/ML Dependencies Check")
    
    ai_packages = [
        ("sentence_transformers", "Text embeddings"),
        ("faiss", "Vector database (faiss-cpu)"),
        ("langchain", "LLM framework"),
        ("langchain_community", "LangChain community"),
        ("langchain_google_genai", "Google Gemini integration"),
        ("google.generativeai", "Google AI SDK"),
        ("huggingface_hub", "HuggingFace model hub")
    ]
    
    results = {}
    
    for package, description in ai_packages:
        try:
            # Handle special cases for import names vs package names
            import_name = package
            if package == "faiss":
                import_name = "faiss"
            elif package == "google.generativeai":
                import_name = "google.generativeai"
            
            mod = importlib.import_module(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {package} ({version}) - {description}")
            results[package] = True
        except ImportError as e:
            print(f"   ❌ {package} - {description}")
            results[package] = False
    
    return results

def check_environment_files():
    """Check for required configuration files"""
    print("\n📄 Configuration Files Check")
    
    files_to_check = [
        (".env.example", "Environment template", False),
        (".env", "Environment configuration", True),
        ("requirements.txt", "Dependencies", False),
        ("requirements-minimal.txt", "Minimal dependencies", False),
        ("src/__init__.py", "Python package structure", False),
        ("src/data_collector.py", "Data collection module", False),
        ("src/vector_store.py", "Vector store module", False),
        ("src/rag_pipeline_gemini.py", "Gemini RAG pipeline", False),
        ("app_gemini.py", "Gemini web app", False),
        ("demo_gemini.py", "Gemini demo script", False),
        ("fix_dependencies.py", "Dependency fix script", False),
        ("app_emergency.py", "Emergency mode app", False)
    ]
    
    all_found = True
    
    for filename, description, required in files_to_check:
        if os.path.exists(filename):
            print(f"   ✅ {filename} - {description}")
        else:
            symbol = "❌" if required else "⚠️"
            print(f"   {symbol} {filename} - {description}")
            if required:
                all_found = False
    
    return all_found

def check_api_keys():
    """Check for API key configuration"""
    print("\n🔑 API Keys Check")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if google_key and google_key != "your_google_api_key_here":
        print("   ✅ Google API key configured")
    else:
        print("   ⚠️  Google API key not configured")
    
    if openai_key and openai_key != "your_openai_api_key_here":
        print("   ✅ OpenAI API key configured")
    else:
        print("   ⚠️  OpenAI API key not configured")
    
    has_keys = (google_key and google_key != "your_google_api_key_here") or \
               (openai_key and openai_key != "your_openai_api_key_here")
    
    return has_keys

def check_sample_data():
    """Check for sample data"""
    print("\n📊 Sample Data Check")
    
    data_files = [
        "data/processed_wikipedia_articles.json",
        "data/demo_articles.json"
    ]
    
    data_found = False
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"   ✅ {data_file}")
            data_found = True
        else:
            print(f"   ⚠️  {data_file} not found")
    
    return data_found

def recommend_next_steps(checks):
    """Provide recommendations based on check results"""
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    if not checks['python']:
        print("🔴 CRITICAL: Upgrade Python to 3.8+")
        return
    
    if not checks['basic_imports']:
        print("🔴 CRITICAL: Python installation is corrupted")
        return
    
    # Check AI dependencies
    ai_issues = sum(1 for k, v in checks['ai_deps'].items() if not v)
    
    if ai_issues >= 3:
        print("🟡 DEPENDENCY CONFLICTS DETECTED")
        print("   Recommended solutions (try in order):")
        print("   1. python fix_dependencies.py")
        print("   2. pip install -r requirements-minimal.txt")
        print("   3. Create clean environment:")
        print("      python -m venv venv_clean")
        print("      source venv_clean/bin/activate")
        print("      pip install -r requirements-minimal.txt")
        print("   4. Use emergency mode: streamlit run app_emergency.py")
    
    elif ai_issues > 0:
        print("🟡 SOME DEPENDENCIES MISSING")
        print("   Try: pip install -r requirements-minimal.txt")
    
    else:
        print("🟢 ALL DEPENDENCIES OK")
    
    if not checks['api_keys']:
        print("\n🔑 API KEYS NEEDED")
        print("   Edit .env file and add your Google API key")
        print("   Get key from: https://console.cloud.google.com/")
    
    if not checks['sample_data']:
        print("\n📊 SAMPLE DATA NEEDED")
        print("   Run: python demo_gemini.py data")
        print("   Or use the web interface data collection")
    
    print("\n🚀 READY TO START:")
    if checks['api_keys'] and ai_issues == 0:
        print("   streamlit run app_gemini.py")
    elif ai_issues > 0:
        print("   streamlit run app_emergency.py  # Safe mode")
    else:
        print("   Fix dependencies first")

def main():
    """Main system check"""
    print("🔍 Wikipedia RAG System - System Check")
    print("="*50)
    
    checks = {
        'python': check_python_version(),
        'basic_imports': check_basic_imports(),
        'key_deps': check_key_dependencies(),
        'ai_deps': check_ai_dependencies(),
        'config_files': check_environment_files(),
        'api_keys': check_api_keys(),
        'sample_data': check_sample_data()
    }
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    total_issues = 0
    if not checks['python']:
        total_issues += 1
        print("❌ Python version incompatible")
    
    if not checks['basic_imports']:
        total_issues += 1
        print("❌ Basic imports failing")
    
    ai_issues = sum(1 for v in checks['ai_deps'].values() if not v)
    if ai_issues > 0:
        total_issues += 1
        print(f"⚠️  {ai_issues} AI dependencies missing/broken")
    
    if not checks['api_keys']:
        total_issues += 1
        print("⚠️  No API keys configured")
    
    if not checks['sample_data']:
        print("ℹ️  No sample data (can be collected)")
    
    if total_issues == 0:
        print("🎉 SYSTEM READY!")
    else:
        print(f"⚠️  {total_issues} issues found")
    
    # Recommendations
    recommend_next_steps(checks)

if __name__ == "__main__":
    main()