from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import List
import os
import logging
import aiohttp
import re
import aiofiles
from utils.text_processing import extract_text, chunk_text
from utils.vector_store import store_chunks, search_similar_chunks
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
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    return file_path

async def process_document(file_path: str, filename: str) -> dict:
    try:
        extracted_text = await extract_text(file_path)
        chunks = await chunk_text(extracted_text)
        index_path = await store_chunks(chunks, {"filename": filename})
        logger.info(f"Document processed: {filename}")
        return {"filename": filename, "file_path": file_path, "num_chunks": len(chunks), "index_path": index_path}
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = await save_uploaded_file(file)
    return await process_document(file_path, file.filename)

@router.post("/upload-multiple")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        file_path = await save_uploaded_file(file)
        result = await process_document(file_path, file.filename)
        results.append(result)
    return results

@router.post("/process-link")
async def process_document_link(document_link: str = Body(..., embed=True)):
    try:
        # Check if it's an arXiv link
        arxiv_id_match = document_link.split('/')[-1]
        if arxiv_id_match:
            document_link = f"https://arxiv.org/pdf/{arxiv_id_match}.pdf"

        async with aiohttp.ClientSession() as session:
            async with session.get(document_link) as response:
                if response.status == 200:
                    content = await response.read()
                    filename = arxiv_id_match + '.pdf'
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(content)
                    return await process_document(file_path, filename)
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to download document")
    except Exception as e:
        logger.error(f"Error processing document link: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document link: {str(e)}")

@router.post("/ask")
async def ask_question(query: str = Body(..., embed=True), index_path: str = Body(..., embed=True)):
    try:
        similar_chunks = await search_similar_chunks(query, index_path)
        answer = await answer_question(query, similar_chunks)
        return {"query": query, "answer": answer}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
