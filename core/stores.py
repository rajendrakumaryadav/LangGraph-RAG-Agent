# core/stores.py
import qdrant_client
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from core import settings


def get_qdrant_client():
    """Returns a Qdrant client instance."""
    return qdrant_client.QdrantClient(url=settings.QDRANT_URL)


def get_vector_store(embeddings):
    """Returns the LangChain Qdrant vector store instance."""
    client = get_qdrant_client()
    
    response = client.get_collections()
    collection_names = [collection.name for collection in response.collections]
    
    if settings.QDRANT_COLLECTION_NAME not in collection_names:
        print(f"Collection '{settings.QDRANT_COLLECTION_NAME}' not found. Creating it now.")
        
        dummy_embedding = embeddings.embed_query("get vector size")
        vector_size = len(dummy_embedding)
        
        client.recreate_collection(collection_name=settings.QDRANT_COLLECTION_NAME,
                                   vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE), )
        print(f"Collection '{settings.QDRANT_COLLECTION_NAME}' created successfully.")
    
    return QdrantVectorStore(client=client, collection_name=settings.QDRANT_COLLECTION_NAME, embedding=embeddings, )
