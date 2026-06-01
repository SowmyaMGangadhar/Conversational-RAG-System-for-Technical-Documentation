from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_chain import RAGPipeline
from src.chat_history import get_session_history

app = FastAPI()

PIPELINES = {}

class ChatRequest(BaseModel):

    query: str

    chunk_type: str = "recursive"
    embedding_type: str = "bge"
    retrieval_type: str = "hybrid"
    reranker_type: str = "bge"
    generation_model: str = "ollama"

    k: int = 5
    temperature: float = 0.2
    session_id: str = "default"


@app.on_event("startup")
def startup_event():

    print("Loading RAG pipeline...")

    pipeline = RAGPipeline(
        chunk_type="recursive",
        embedding_type="bge",
        retrieval_type="hybrid",
        reranker_type="bge",
        generation_model="ollama"
    )

    PIPELINES["default"] = pipeline

    print("RAG pipeline loaded successfully")



# @app.post("/chat")
# def chat(request: ChatRequest):

#     pipeline = PIPELINES["default"]

#     result = pipeline.run(
#         query=request.query,
#         k=request.k,
#         temperature=request.temperature
#     )

#     return result

@app.post("/chat")
def chat(request: ChatRequest):

    pipeline_key = (
        f"{request.chunk_type}_"
        f"{request.embedding_type}_"
        f"{request.retrieval_type}_"
        f"{request.reranker_type}_"
        f"{request.generation_model}")
    
    if pipeline_key not in PIPELINES:

        print(f"Creating new pipeline: {pipeline_key}")

        PIPELINES[pipeline_key] = RAGPipeline(

            chunk_type=request.chunk_type,
            embedding_type=request.embedding_type,
            retrieval_type=request.retrieval_type,
            reranker_type=request.reranker_type,
            generation_model=request.generation_model)

    pipeline = PIPELINES[pipeline_key]

    history = get_session_history(request.session_id)

    result = pipeline.run(
        query=request.query,
        k=request.k,
        temperature=request.temperature,
        session_history = history
    )

    return result