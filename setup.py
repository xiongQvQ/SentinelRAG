#!/usr/bin/env python3
"""
Setup script for Wikipedia RAG System Demo
Helps users get started quickly with the system
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def install_dependencies():
    """Install required dependencies with fallback options"""
    print("\n📦 Installing dependencies...")
    
    # Try minimal requirements first (more stable)
    try:
        if os.path.exists("requirements-minimal.txt"):
            print("Trying minimal requirements (recommended)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-minimal.txt"])
            print("✅ Minimal dependencies installed successfully")
            return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Minimal requirements failed: {e}")
    
    # Fallback to full requirements
    try:
        print("Trying full requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Full dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("\n💡 Recommendation: Run the dependency fix script:")
        print("   python fix_dependencies.py")
        return False

def setup_environment():
    """Set up environment file"""
    print("\n⚙️ Setting up environment...")
    
    env_file = Path(".env")
    example_file = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if example_file.exists():
        # Copy example file
        with open(example_file, 'r') as src:
            content = src.read()
        with open(env_file, 'w') as dst:
            dst.write(content)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and add your Google API key")
        return True
    else:
        # Create basic .env file
        with open(env_file, 'w') as f:
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print("✅ Created basic .env file")
        print("⚠️  Please edit .env file and add your API keys")
        return True

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API keys...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if google_key and google_key != "your_google_api_key_here":
        print("✅ Google API key configured")
        return True
    elif openai_key and openai_key != "your_openai_api_key_here":
        print("✅ OpenAI API key configured")
        return True
    else:
        print("⚠️  No API keys configured")
        print("Please edit .env file and add your Google API key (recommended) or OpenAI API key")
        return False

def test_basic_imports():
    """Test basic imports"""
    print("\n🧪 Testing imports...")
    
    try:
        from src.data_collector import WikipediaDataCollector
        print("✅ Data collector import successful")
        
        from src.vector_store import VectorStoreManager
        print("✅ Vector store import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. 🔑 Make sure your API keys are set in .env file")
    print("2. 🚀 Run the demo:")
    print("   python demo_gemini.py      # For Google Gemini (recommended)")
    print("   python demo.py             # For OpenAI")
    print("3. 🌐 Or start the web interface:")
    print("   streamlit run app_gemini.py  # For Google Gemini (recommended)")
    print("   streamlit run app.py         # For OpenAI")
    print("\n📚 Available commands:")
    print("   python demo_gemini.py test    # Quick system test")
    print("   python demo_gemini.py data    # Collect sample data")
    print("   python demo_gemini.py compare # Compare models")
    print("\n📖 See README.md for detailed documentation")

def main():
    """Main setup function"""
    print("🚀 Wikipedia RAG System Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Check API keys
    api_keys_configured = check_api_keys()
    
    # Test imports
    if not test_basic_imports():
        return False
    
    # Display next steps
    display_next_steps()
    
    if not api_keys_configured:
        print("\n⚠️  Remember to configure your API keys before running the demo!")
    
    return True

if __name__ == "__main__":
    main()