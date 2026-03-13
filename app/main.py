from fastapi import FastAPI
from app.schemas import QueryRequest, QueryResponse
from app.rag_pipeline import qa_chain

app = FastAPI(title="Research Paper RAG API", version="1.0")

@app.post("/query", response_model=QueryResponse)
def query_paper(request: QueryRequest):
    result = qa_chain.invoke({"input": request.question})
    
    sources = [doc.metadata for doc in result.get("context", [])]
    
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources
    )
