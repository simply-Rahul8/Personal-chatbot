import streamlit as st
import subprocess
import sys

def test_ollama_connection():
    """Test Ollama connection"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Ollama is running and models are available")
            st.code(result.stdout)
        else:
            st.error("❌ Ollama connection failed")
    except FileNotFoundError:
        st.error("❌ Ollama not found. Please install Ollama first.")

def test_dependencies():
    """Test if all dependencies are installed"""
    required_packages = [
        'streamlit', 'langchain', 'langchain-community', 
        'langchain-ollama', 'faiss-cpu', 'pypdf2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            st.success(f"✅ {package}")
        except ImportError:
            st.error(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"Install missing packages: pip install {' '.join(missing_packages)}")

if __name__ == "__main__":
    st.title("🧪 System Test")
    
    st.header("Dependencies Check")
    test_dependencies()
    
    st.header("Ollama Connection Test")
    test_ollama_connection()
