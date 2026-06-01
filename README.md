# Conversational-RAG-System-for-Technical-Documentation

An end-to-end conversational RAG system built for querying technical documentation across FastAPI, LangChain, PyTorch, and NVIDIA TensorRT. The project uses modular pipelines for document ingestion, chunking, embeddings, retrieval, reranking, and response generation to produce grounded answers from documentation sources.

The system supports configurable retrieval strategies, local LLM inference, evaluation using retrieval and RAGAS metrics, and an interactive FastAPI + Streamlit interface for experimentation and testing.


It supports:

Multi-source documentation ingestion
Recursive, markdown, semantic, and parent-child chunking
Multiple embedding backends such as BGE, E5, Instructor, and Nomic
FAISS vector search
Hybrid retrieval using BM25 + dense vector search
BGE reranking
Local LLM inference with Ollama
FastAPI backend
Streamlit chat interface
Conversational memory
Retrieval and generation evaluation using Precision@K, Recall@K, HitRate@K, and RAGAS

## Tech Stack
Python
LangChain
FAISS
BM25
BGE Embeddings
BGE Reranker
Ollama
FastAPI
Streamlit
RAGAS
Project Structure

src/
├── ingestion/
├── cleaning/
├── chunking/
├── embedding/
├── vectorstore/
├── retrieval/
├── reranking/
├── generation/
├── prompts/
├── evaluation/
├── rag_chain.py
├── app.py
├── streamlit_app.py
└── main.py

## How to Run
1. Create virtual environment
python3 -m venv venv
source venv/bin/activate
2. Install dependencies
pip install -r requirement.txt
3. Start Ollama

Install Ollama and pull a local model:

ollama pull qwen2.5:7b

Make sure Ollama is running.

4. Run FastAPI backend

From project root:

uvicorn src.app:app --reload

FastAPI will run at:

http://127.0.0.1:8000
5. Run Streamlit frontend

Open a second terminal:

streamlit run src/streamlit_app.py
Evaluation

The project includes retrieval and generation evaluation.

Retrieval metrics:

Precision@K
Recall@K
HitRate@K

Generation metrics using RAGAS:

Faithfulness
Answer Relevancy

Current benchmark results:

Faithfulness: 1.0
Answer Relevancy: ~0.70
HitRate@5: 1.0