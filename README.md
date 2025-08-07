# Personal-chatbot

Personal Data Chatbot with Streamlit, RAG, Ollama, and LangChain
A privacy-preserving chatbot that allows you to interact with your personal documents (PDF or TXT) using Retrieval-Augmented Generation (RAG). The chatbot loads your documents locally, indexes their content with embeddings, and answers your questions using an Ollama-hosted large language model (LLM). This project uses Streamlit for the web interface and LangChain for orchestrating the RAG pipeline.

Features
Upload PDF or TXT documents containing your personal data.

Process and index the entire document without filtering out details.

Query and receive answers based solely on your uploaded document.

Local language model inference via Ollama (no cloud dependencies).

Convenient Streamlit web UI with improved file uploader styling.

Green dot indicator confirms successful document processing.

Multi-turn chat history with session state.

Simple, clean UI with enlarged fonts and margin adjustment.

Fixed model "deepseek-r1:1.5b" for consistent inference.

Easily extendable for other Ollama models or file types.

Requirements
Python 3.9+

Ollama CLI and daemon installed and running

At least 8GB RAM recommended (16GB+ preferred for larger models)

Supported OS: macOS, Linux, Windows

Installation
Clone the repository

Create and activate a virtual environment

bash
python -m venv rag_env
.\rag_env\Scripts\activate   # Windows
source rag_env/bin/activate  # macOS/Linux
Install dependencies

bash
pip install -r requirements.txt
If you don't have a requirements.txt yet, manually install:

bash
pip install streamlit langchain langchain-ollama langchain-community faiss-cpu pypdf2
Install and set up Ollama

Download Ollama from https://ollama.com and install on your machine.

Pull the required Ollama model:

bash
ollama pull deepseek-r1:1.5b
Verify model availability:

bash
ollama list
Ensure Ollama daemon is running (usually runs in background automatically).

Usage
Start the Streamlit app:

bash
streamlit run app.py
Open the local URL provided by Streamlit in your web browser.

Upload your PDF or TXT document using the file uploader (aligned with margin).

Wait until you see the green dot and "Document processed!" message.

In the input box below, ask questions related to your document content, including personal information.

Interact with the chatbot and get replies based on your document.

Use the "Clear Conversation" button to reset chat history at any time.

File Uploads
Supports PDF and plain text (.txt) files.

The file uploader is styled with a margin shift to the right for better UI.

Whole documents are processed — no filtering of any detail.

Customization
To change the Ollama model, modify OLLAMA_MODEL in app.py (make sure the model is pulled locally).

Adjust text chunk size and overlap in CHUNK_SIZE and CHUNK_OVERLAP constants.

Modify prompt behavior in the PROMPT_TEMPLATE string.

Extend to other document types by adding loaders.

Troubleshooting
Model not found error: Ensure you have pulled the Ollama model locally using:

bash
ollama pull deepseek-r1:1.5b
Ollama API errors: Verify Ollama daemon is running and accessible.

File type not supported: Only PDF and TXT file uploads are supported.

Memory issues: Use smaller or quantized models or reduce chunk sizes.

Streamlit interface issues: Clear Streamlit cache with:

bash
streamlit cache clear
Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements and bug fixes.

License
Specify your license here, e.g., MIT License.

Acknowledgements
Streamlit

LangChain

Ollama

Inspired by open-source RAG chatbot architectures.
