import streamlit as st
import tempfile
import os
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGChain

# Page configuration
st.set_page_config(
    page_title="Personal RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def process_uploaded_files(uploaded_files):
    """Process uploaded files and create vector store"""
    if not uploaded_files:
        return False
    
    # Save uploaded files temporarily
    temp_files = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_files.append(tmp_file.name)
    
    try:
        # Process documents
        processor = DocumentProcessor()
        documents = processor.load_documents(temp_files)
        
        if not documents:
            st.error("No documents could be processed.")
            return False
        
        # Create vector store
        vector_manager = VectorStoreManager()
        vector_store = vector_manager.create_faiss_store(documents)
        
        # Setup RAG chain
        rag_chain = RAGChain()
        retriever = vector_manager.get_retriever(k=3)
        rag_chain.setup_chain(retriever)
        
        # Store in session state
        st.session_state.rag_chain = rag_chain
        st.session_state.vector_store = vector_store
        
        st.success(f"Processed {len(documents)} document chunks successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def main():
    st.title("🤖 Personal RAG Chatbot")
    st.write("Upload your documents and chat with them using local AI!")
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    success = process_uploaded_files(uploaded_files)
                    if success:
                        st.rerun()
            else:
                st.warning("Please upload some files first.")
        
        
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.rag_chain:
            st.error("Please upload and process documents first!")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.query(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
