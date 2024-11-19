from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, conint
from milvus_helper import query_by_embedding
from ollama_helper import generate_answer_with_ollama

# Create a FastAPI router
router = APIRouter()


# Define the request model
class QueryRequest(BaseModel):
    user_query: str
    limit: conint(ge=1, le=10) = Field(default=3, description="Limit the number of results (between 1 and 10)")
    require_ip: bool = Field(default=False, description="Whether an IP address is required")
    ip_address: str = None  # Optional field to hold IP address if required

    class Config:
        orm_mode = True
        validate_assignment = True


# Define the endpoint for querying and generating answers
@router.post("/query", status_code=200)
async def query_and_generate_answer(request: QueryRequest):
    try:
        # Check if IP is required but not provided
        if request.require_ip and not request.ip_address:
            raise HTTPException(status_code=500, detail="No relevant context found to generate an answer.")

        # Use the helper functions to get context and generate an answer
        context = query_by_embedding(request.user_query, limit=request.limit)

        if not context:
            raise HTTPException(status_code=404, detail="No relevant context found to generate an answer.")

        # Generate the answer using the context
        answer = generate_answer_with_ollama(request.user_query, context)

        if not answer:
            raise HTTPException(status_code=500, detail="Failed to generate an answer.")

        return {"answer": answer}

    except HTTPException as http_exc:
        # Re-raise any HTTP exceptions
        raise http_exc

    except Exception as exc:
        # Catch any other exceptions and return a 500 error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(exc)}")
