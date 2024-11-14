from typing import List
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import HTTPException

# async def extract_text(file_path: str) -> str:
#     elements = partition(filename=file_path)
#     return "\n\n".join([str(el) for el in elements])

async def extract_text(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)  # Open the PDF
        all_text = ""
        for page in doc:
            all_text += page.get_text("text") + "\n"
        doc.close()
        return all_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

async def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)