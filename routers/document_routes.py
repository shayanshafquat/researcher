from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import List
import os
from utils.text_processing import extract_text, chunk_text
from utils.vector_store import store_chunks
from utils.rag_pipeline import answer_question

router = APIRouter()

UPLOAD_DIR = "uploads"

async def save_uploaded_file(file: UploadFile) -> str:
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    return file_path

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = await save_uploaded_file(file)
    extracted_text = await extract_text(file_path)
    chunks = await chunk_text(extracted_text)
    await store_chunks(chunks, {"filename": file.filename})
    return {"filename": file.filename, "file_path": file_path, "num_chunks": len(chunks)}

@router.post("/upload-multiple")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        file_path = await save_uploaded_file(file)
        extracted_text = await extract_text(file_path)
        chunks = await chunk_text(extracted_text)
        await store_chunks(chunks, {"filename": file.filename})
        uploaded_files.append({
            "filename": file.filename,
            "file_path": file_path,
            "num_chunks": len(chunks)
        })
    return uploaded_files

@router.post("/ask")
async def ask_question(query: str = Body(..., embed=True)):
    answer = await answer_question(query)
    return {"query": query, "answer": answer}