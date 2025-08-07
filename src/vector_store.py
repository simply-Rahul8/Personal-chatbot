from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings as OllamaEmbeddingsNew


class VectorStoreManager:
    def __init__(self, embedding_model: str = "llama3.2"):
        # Use Ollama for embeddings
        self.embeddings = OllamaEmbeddingsNew(model=embedding_model)
        self.vector_store = None
    
    def create_faiss_store(self, documents: List[Document]):
        """Create FAISS vector store from documents"""
        self.vector_store = FAISS.from_documents(
            documents, 
            self.embeddings
        )
        return self.vector_store
    
    def create_chroma_store(self, documents: List[Document], persist_directory: str = "./chroma_db"):
        """Create Chroma vector store from documents"""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return self.vector_store
    
    def save_faiss_store(self, path: str):
        """Save FAISS store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
    
    def load_faiss_store(self, path: str):
        """Load FAISS store from disk"""
        self.vector_store = FAISS.load_local(path, self.embeddings)
        return self.vector_store
    
    def get_retriever(self, k: int = 4):
        """Get retriever for similarity search"""
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        return None
