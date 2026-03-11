# 👨‍🍳 Smart Arabic Chef: Full-Stack RAG Application

An end-to-end full-stack AI application that acts as a professional chef. Users can ask for recipes using everyday Arabic, and the AI translates the query, retrieves the exact ingredients and steps from a custom-built local database, and generates a precise, hallucination-free response in Modern Standard Arabic (MSA).

This project demonstrates the complete lifecycle of a **Retrieval-Augmented Generation (RAG)** system, from web scraping and vector database indexing to building a FastAPI backend and a Gradio frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)

---

## ✨ Features
* **Custom Dataset Construction:** Includes a web scraper built with BeautifulSoup to gather authentic Arabic recipes from sitemaps.
* **Dialect Translation Pipeline:** Uses an LLM to translate colloquial Egyptian Arabic queries into Modern Standard Arabic for highly accurate vector similarity search.
* **Strict Hallucination Prevention:** The generation prompt is strictly engineered to only use context from the retrieved recipes and apologize if a recipe is not found.
* **Local Vector Database:** Uses HuggingFace Embeddings and ChromaDB to process and store text embeddings entirely locally.
* **Decoupled Architecture:** Clean separation of concerns with a FastAPI backend and a Gradio frontend communicating via REST API.

---

## 📂 Project Structure

📁 RAG/
├── 📁 Scrapping and indexing/
│   ├── web_scraping.py       # Scrapes recipes and saves to JSONL
│   ├── indexing.py           # Embeds JSONL into local ChromaDB
│   └── recipe_vector_db/     # (Generated) Local Chroma vector store
├── 📁 notebooks/             # Jupyter notebooks for prototyping and testing
├── app.py                    # FastAPI backend & LangChain RAG pipeline
├── frontend.py               # Gradio chat interface UI
├── run_app.py                # Master script to launch backend & frontend together
└── .env                      # (Ignored) Contains API keys

---

## 🚀 Tech Stack
* **LLM:** Llama-3.3-70b-versatile (via Groq API)
* **Embeddings:** HuggingFace (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
* **Vector Database:** ChromaDB
* **Backend:** FastAPI, LangChain, Uvicorn
* **Frontend:** Gradio, Requests
* **Data Engineering:** BeautifulSoup4, Requests

---

## ⚙️ Installation & Setup

**1. Clone the repository:**
git clone https://github.com/YOUR_USERNAME/smart-chef-rag.git
cd smart-chef-rag

**2. Install dependencies:**
pip install fastapi uvicorn gradio langchain langchain-groq langchain-huggingface langchain-community chromadb sentence-transformers python-dotenv requests beautifulsoup4 lxml

**3. Set up environment variables:**
Create a file named .env in the root directory and add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

**4. Build the Local Vector Database:**
Because the vector database is too large for GitHub, you need to build it locally first. Ensure you have the scraped jsonl file, then run:
python "Scrapping and indexing/indexing.py"

*(This will download the embedding model to a local_models folder and create the recipe_vector_db folder).*

---

## 🏃‍♂️ How to Run the Application

The easiest way to run the entire application is using the included launcher script. It will automatically start the FastAPI server, wait for the AI models to load into memory, launch the Gradio UI, and open your web browser.

**Run the master script:**
python run_app.py

### Manual Startup (Alternative)
If you prefer to run the services separately:

1. **Start the Backend:**
uvicorn app:app
*(Wait for the terminal to say "Models loaded successfully. Server ready.")*

2. **Start the Frontend:** Open a second terminal and run:
python frontend.py
