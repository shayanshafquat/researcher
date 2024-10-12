import os
from typing import List
from langchain.vectorstores import FAISS
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

index_file = "faiss_index.bin"

def create_or_load_index(chunks: List[str], index_file: str):
    if os.path.exists(index_file):
        index = FAISS.load_local(index_file, embeddings)
    else:
        index = FAISS.from_texts(chunks, embeddings)
        index.save_local(index_file)
    return index

async def store_chunks(chunks: List[str], metadata: dict):
    index = create_or_load_index(chunks, index_file)
    index.save_local(index_file)

async def search_similar_chunks(query: str, k: int = 5):
    index = FAISS.load_local(index_file, embeddings)
    similar_chunks = index.similarity_search(query, k=k)
    return similar_chunks