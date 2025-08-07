import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process PDF files"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_text(self, file_path: str) -> List[Document]:
        """Load and process text files"""
        loader = TextLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load multiple documents of different types"""
        all_documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                documents = self.load_pdf(file_path)
            elif file_path.endswith('.txt'):
                documents = self.load_text(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            
            all_documents.extend(documents)
        
        return all_documents
