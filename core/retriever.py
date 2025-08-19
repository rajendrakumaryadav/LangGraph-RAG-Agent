# core/retriever.py
from langchain.retrievers import (ContextualCompressionRetriever, MultiQueryRetriever, )
from langchain.retrievers.document_compressors import LLMChainExtractor

from core.models import get_generator_model


def create_retriever(vector_store):
    """
    Creates an advanced retriever with MultiQuery, MMR, and Contextual Compression.
    """
    base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 8, 'fetch_k': 20})
    
    llm = get_generator_model()
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                           base_retriever=multi_query_retriever)
    
    return compression_retriever
