from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import document_routes
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_routes.router, prefix="/documents", tags=["documents"])

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG-based application"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)