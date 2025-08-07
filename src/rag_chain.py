from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama  # ✅ correct
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGChain:
    def __init__(self, model_name: str = "llama3.2"):
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.retriever = None
        self.chain = None
        
        # Define the prompt template
        self.prompt_template = """
        You are a helpful AI assistant. Use the following context to answer the user's question. 
        If you don't know the answer based on the context, say so clearly.
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def setup_chain(self, retriever):
        """Setup the RAG chain with retriever"""
        self.retriever = retriever
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the RAG chain
        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self.chain
    
    def query(self, question: str) -> str:
        """Query the RAG chain"""
        if not self.chain:
            return "RAG chain not initialized. Please setup the chain first."
        
        try:
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"Error processing query: {str(e)}"
