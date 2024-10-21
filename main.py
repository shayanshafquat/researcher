import uvicorn
from fastapi import FastAPI
from routers import document_routes

app = FastAPI()
app.include_router(document_routes.router, prefix="/documents")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
