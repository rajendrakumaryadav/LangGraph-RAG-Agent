# app_streamlit.py
import os
import sys
import streamlit as st
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ingest import ingest_paths
from core.rag import create_rag_workflow

st.set_page_config(page_title="Advanced RAG System", page_icon="🧠", layout="wide")

# State Management
if "rag_app" not in st.session_state:
    st.session_state.rag_app = create_rag_workflow()

if "messages" not in st.session_state:
    st.session_state.messages = []

# UI Layout
st.title("🧠 Advanced RAG with LangGraph")

# Sidebar for document ingestion
with st.sidebar:
    st.header("1. Ingest Documents")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"],
                                      accept_multiple_files=True)
    if st.button("Ingest Documents"):
        if uploaded_files:
            temp_dir = "uploads"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            file_paths = [os.path.join(temp_dir, f.name) for f in uploaded_files]
            for file, path in zip(uploaded_files, file_paths):
                with open(path, "wb") as f_out:
                    f_out.write(file.read())
            
            with st.spinner("Ingesting documents..."):
                ingest_paths(file_paths)
                st.success("✅ Ingestion complete!")
            
            for path in file_paths:
                os.remove(path)
        else:
            st.warning("Please upload at least one document.")

st.divider()

# Main chat interface
st.header("2. Ask Questions")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            result = st.session_state.rag_app.invoke({"question": prompt})
            response = result.get("final_answer", "Sorry, I couldn't find an answer.")
            message_placeholder.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
