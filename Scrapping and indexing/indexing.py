import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. Load Preprocessed Data ---
documents = []

# Load dataset from the same directory as the script
with open('rag_recipes_dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        recipe = json.loads(line)
        
        # Structure data into LangChain Document objects
        doc = Document(
            page_content=recipe['text_for_embedding'],
            metadata={"title": recipe['title'], "url": recipe['url']}
        )
        documents.append(doc)

print(f"Loaded {len(documents)} recipes into memory.")

# --- 2. Initialize Embedding Model ---
print("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}  # Change to 'cuda' if you have a local NVIDIA GPU
)

# --- 3. Build and Persist Vector Database ---
print("Generating embeddings and building ChromaDB...")

# Create the vector store and save it in the current folder
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./recipe_vector_db"
)

print("Indexing Complete. Vector database successfully persisted.")