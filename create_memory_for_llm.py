import os
import sys
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Ensure UTF-8 printing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(find_dotenv())

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def create_vector_db():
    print(f"Loading raw PDF files from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} document pages.")
    
    # Optimal chunk size & overlap determined by empirical benchmark
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    
    # Enrich metadata for each chunk (source, page, topic, section)
    for chunk in text_chunks:
        src = os.path.basename(chunk.metadata.get("source", "Medical Reference"))
        chunk.metadata["source"] = src
        chunk.metadata["page"] = chunk.metadata.get("page_label", chunk.metadata.get("page", 1))
        
        # Extract section / topic title from first line of chunk text
        lines = chunk.page_content.strip().split("\n")
        first_line = lines[0].strip() if lines else ""
        if len(first_line) > 5 and len(first_line) < 80:
            chunk.metadata["section"] = first_line
            chunk.metadata["topic"] = first_line
        else:
            chunk.metadata["section"] = "Clinical Reference"
            chunk.metadata["topic"] = src.replace(".pdf", "").replace("_", " ").title()
            
    print(f"Created {len(text_chunks)} chunks with enriched metadata (source, page, topic, section).")
    
    print("Initializing HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    print("Building FAISS VectorStore index with metadata...")
    db = FAISS.from_documents(text_chunks, embedding_model)
    
    os.makedirs("vectorstore", exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS VectorStore successfully created and saved to {DB_FAISS_PATH}")


if __name__ == "__main__":
    create_vector_db()