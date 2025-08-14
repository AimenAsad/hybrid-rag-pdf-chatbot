# Chat with PDF (Hybrid RAG using LangChain, Ollama & FAISS)

A local **"Chat with PDF"** application that lets you query multiple PDFs using **Hybrid Retrieval-Augmented Generation (RAG)** combining **semantic search** (FAISS vector embeddings) with **keyword search** for better accuracy.  

No paid APIs needed — runs entirely **offline** with [Ollama](https://ollama.ai) and your own PDFs.

---

## Features
**Multiple PDF Support** – Load any number of PDFs from a folder  
**Hybrid Retrieval** – Combines semantic search with keyword matching  
**Local LLM & Embeddings** – Uses Ollama (`llama3:instruct` + `nomic-embed-text`)  
**Source Metadata** – Retrieves context with PDF file name & page number  
**Web UI** – Built with Gradio for easy interaction  
**Runs 100% Locally** – No internet or paid APIs required  

---

## 🖼 Tech Architecture
PDFs → Loader (PyMuPDF) → Text Chunking → Embeddings (Ollama) → FAISS Vector Store ↔ Hybrid Retriever ↔ LLM (Ollama) → Answer

---

## Project Structure

RAG/
│── app.py # Gradio Web App<br>
│── main.py # CLI test script<br>
│── vector_embedding.py # Vector store builder<br>
│── hybrid_retriever.py # Hybrid (semantic + keyword) search<br>
│── retriever.py # FAISS-only retriever<br>
│── pdf_loader.py # Loads PDFs from folder<br>
│── chunking.py # Splits text into chunks<br>
│── RAG-dataset/ # Your PDFs<br>
│── faiss_index/ # Saved FAISS index<br>
│── requirements.txt<br>
│── README.md


---

## Installation & Setup
### 1️ Clone Repository

git clone <https://github.com/AimenAsad/hybrid-rag-pdf-chatbot.git>

### 2️ Create Virtual Environment

python3 -m venv rag_env
source rag_env/bin/activate

### 3️ Install Dependencies

pip install -r requirements.txt

### 4️ Install & Pull Ollama Models

ollama pull llama3:instruct
ollama pull nomic-embed-text

### 5️ Add PDFs

Place your PDFs in the RAG-dataset/ folder.

### 6️ Build Vector Store

python3 vector_embedding.py

### 7️ Run Web App

python3 app.py

---

## Example Questions

Top three most consumed supplements<br>
What are side effects of creatine?<br>
How to gain muscle mass effectively?<br>
Which supplements help with fat loss?

## How It Works

PDF Loader – Loads PDFs and extracts text using PyMuPDF<br>
Chunking – Splits long text into smaller chunks for better retrieval<br>
Embedding – Converts chunks into vectors using nomic-embed-text model<br>
FAISS Vector Store – Stores embeddings for fast similarity search<br>
Hybrid Retrieval – Combines:<br>
Semantic Search → Finds context by meaning<br>
Keyword Search → Matches exact terms from query<br>
LLM Response – Sends retrieved chunks to llama3:instruct for answer generation

## Why Hybrid Retrieval?

Semantic Search sometimes misses obvious keyword matches<br>
Keyword Search can't handle synonyms or rephrasing<br>
Hybrid ensures higher recall + better accuracy
