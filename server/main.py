from fastapi import FastAPI
from api import router as api_router

# Initialize FastAPI
app = FastAPI()

# Include the API router from api.py
app.include_router(api_router)
