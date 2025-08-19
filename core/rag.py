# core/rag.py
import json
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from core.models import get_generator_model, get_critic_model, get_embedding_model
from core.retriever import create_retriever
from core.stores import get_vector_store


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    final_answer: str


def retrieve_documents(state: GraphState) -> GraphState:
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    embeddings = get_embedding_model()
    vector_store = get_vector_store(embeddings)
    retriever = create_retriever(vector_store)
    documents = retriever.invoke(question)
    print(documents)
    return {"documents": documents, "question": question}


def generate_answer(state: GraphState) -> GraphState:
    print("---NODE: GENERATE INITIAL ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an expert Q&A assistant. Use the following context to answer the user's question. "
                    "If you don't know the answer, just say that you don't know. Be concise and helpful.\n\n"
                    "CONTEXT:\n{context}"), ("human", "Question: {question}")])
    
    llm = get_generator_model()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = {"context": lambda x: format_docs(x["documents"]), "question": lambda x: x["question"]} | prompt | llm
    generation = rag_chain.invoke({"documents": documents, "question": question}).content
    print(f"Generated answer: {generation}")
    return {"generation": generation}


def critique_answer(state: GraphState) -> GraphState:
    print("---NODE: CRITIQUE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a meticulous fact-checker. Evaluate the generated answer based on the provided context. "
                    "Provide a JSON object with two keys:\n"
                    "- 'decision': Either 'accept' or 'revise'.\n"
                    "- 'revision': If 'revise', provide a corrected answer. If 'accept', this can be an empty string.\n\n"
                    "CONTEXT:\n{context}"),
         ("human", "Question: {question}\nGenerated Answer: {generation}\n\nYour JSON evaluation:")])
    
    critic_llm = get_critic_model()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    critic_chain = {"context": lambda x: format_docs(x["documents"]), "question": lambda x: x["question"],
                    "generation": lambda x: x["generation"]} | prompt | critic_llm
    
    critique_output = critic_chain.invoke(state).content
    
    try:
        critique_json = json.loads(critique_output)
        if critique_json.get("decision") == "accept":
            final_answer = generation
        else:
            final_answer = critique_json.get("revision", generation)
    except json.JSONDecodeError:
        final_answer = generation
    
    source_docs = [doc.metadata.get('source', 'Unknown') for doc in documents]
    unique_sources = sorted(list(set(source_docs)))
    final_answer_with_sources = f"{final_answer}\n\n**Sources:**\n- " + "\n- ".join(unique_sources)
    
    return {"final_answer": final_answer_with_sources}


def create_rag_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("critique", critique_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critique")
    workflow.add_edge("critique", END)
    return workflow.compile()
