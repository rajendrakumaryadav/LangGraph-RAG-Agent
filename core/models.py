# core/models.py
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings

from core import settings


def get_embedding_model():
    """Returns the embedding model instance."""
    return OllamaEmbeddings(model=settings.EMBEDDING_MODEL)


def get_generator_model():
    """Returns the generator model instance."""
    return ChatOllama(model=settings.GENERATOR_MODEL, temperature=0)


def get_critic_model():
    """Returns the critic model instance."""
    return ChatGroq(temperature=0, model_name=settings.CRITIC_MODEL, api_key=settings.GROQ_API_KEY)
