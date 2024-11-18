from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from milvus_helper import query_by_embedding
from ollama_helper import generate_answer_with_ollama

# Create a FastAPI router
router = APIRouter()

# Define the request model
class QueryRequest(BaseModel):
    user_query: str
    limit: int = 5

# Define the endpoint for querying and generating answers
@router.post("/query")
async def query_and_generate_answer(request: QueryRequest):
    # Use the helper functions to get context and generate an answer
    context = query_by_embedding(request.user_query, limit=request.limit)
    if context:
        answer = generate_answer_with_ollama(request.user_query, context)
        if answer:
            return {"answer": answer}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate an answer.")
    else:
        raise HTTPException(status_code=404, detail="No relevant context found to generate an answer.")
