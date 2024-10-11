from typing import List
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter

async def extract_text(file_path: str) -> str:
    elements = partition(filename=file_path)
    return "\n\n".join([str(el) for el in elements])

async def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)