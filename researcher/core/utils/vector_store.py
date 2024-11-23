import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

INDEX_DIR = "faiss_indexes"

def ensure_index_dir():
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

def create_index(chunks: List[str], metadata: Dict[str, str]):
    ensure_index_dir()
    
    # Create a new index for the current document
    new_index = FAISS.from_texts(chunks, embeddings, metadatas=[metadata] * len(chunks))
    
    # Generate a unique filename for this document
    index_filename = f"index_{metadata['filename'].replace('.', '_')}.bin"
    index_path = os.path.join(INDEX_DIR, index_filename)
    
    # Save the new index
    new_index.save_local(index_path)
    return index_path

async def store_chunks(chunks: List[str], metadata: Dict[str, str]):
    index_path = create_index(chunks, metadata)
    return index_path

async def search_similar_chunks(query: str, index_path: str, k: int = 5):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index found at {index_path}. Please upload a document first.")
    
    index = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    similar_chunks = index.similarity_search(query, k=k)
    return similar_chunks
