import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from researcher.core.routers import document_routes

app = FastAPI(
    title="ResearchGPT API",
    description="Backend API for ResearchGPT: Your AI Research Assistant for Scientific Papers",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_routes.router, prefix="/documents")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
