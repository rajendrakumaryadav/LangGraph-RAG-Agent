# core/ingest.py
from typing import List
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.models import get_embedding_model
from core.stores import get_vector_store

def ingest_paths(paths: List[str]):
    """
    Ingests documents from a list of file paths into the Qdrant vector store.
    """
    print(f"Ingesting {len(paths)} document(s)...")
    
    all_docs = []
    for path in paths:
        try:
            loader = DoclingLoader(path)
            docs = loader.load()
            print(f"  - Loaded document(s) from {path} using Docling.")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  - Failed to load {path}: {e}")
            continue

    if not all_docs:
        print("No documents were loaded. Aborting ingestion.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(chunks)} chunks.")

    embeddings = get_embedding_model()
    # The 'embeddings' object is now primarily used for the collection check
    vector_store = get_vector_store(embeddings)
    
    # --- FIX: Remove the 'embedding' argument, only pass batch_size ---
    vector_store.add_documents(chunks, batch_size=32)
    print(f"Successfully ingested {len(chunks)} chunks into Qdrant.")