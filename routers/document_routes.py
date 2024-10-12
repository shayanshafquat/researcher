from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import List
import os
import logging
from utils.text_processing import extract_text, chunk_text
from utils.vector_store import store_chunks
from utils.rag_pipeline import answer_question

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    return file_path

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = await save_uploaded_file(file)
        extracted_text = await extract_text(file_path)
        chunks = await chunk_text(extracted_text)
        logger.info(f"Document chunking done: {file.filename}")
        await store_chunks(chunks, {"filename": file.filename})
        logger.info(f"Document uploaded and processed: {file.filename}")
        return {"filename": file.filename, "file_path": file_path, "num_chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

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
