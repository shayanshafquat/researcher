import os
from typing import List
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

INDEX_FILE = "faiss_index.bin"

def create_or_load_index(dimension: int):
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(dimension)
    return index

def save_index(index):
    faiss.write_index(index, INDEX_FILE)

async def store_chunks(chunks: List[str], metadata: dict):
    vectors = [embeddings.embed(chunk) for chunk in chunks]
    index = create_or_load_index(len(vectors[0]))
    index.add(np.array(vectors, dtype=np.float32))
    save_index(index)

async def search_similar_chunks(query: str, k: int = 5):
    query_vector = np.array([embeddings.embed(query)], dtype=np.float32)
    index = create_or_load_index(query_vector.shape[1])
    distances, indices = index.search(query_vector, k)
    return indices