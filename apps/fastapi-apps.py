# app_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from core.rag import create_rag_workflow

# Pydantic models for request and response
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Dictionary to hold our RAG app instance
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the RAG workflow on startup
    print("Loading RAG workflow...")
    state["rag_app"] = create_rag_workflow()
    print("RAG workflow loaded.")
    yield
    # Clean up resources if needed on shutdown
    state.clear()
    print("RAG workflow cleared.")

app = FastAPI(lifespan=lifespan)

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Endpoint to ask a question to the RAG system.
    """
    if "rag_app" not in state:
        raise HTTPException(status_code=503, detail="RAG workflow is not available")
    
    try:
        print(f"Received question: {request.question}")
        result = state["rag_app"].invoke({"question": request.question})
        answer = result.get("final_answer", "Could not process the request.")
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"Error during RAG invocation: {e}")
        raise HTTPException(status_code=500, detail="Error processing the question.")

@app.get("/")
def read_root():
    return {"message": "Advanced RAG API is running. Use the /ask endpoint to post questions."}