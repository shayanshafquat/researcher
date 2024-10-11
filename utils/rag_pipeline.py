import openai
import os
from utils.vector_store import search_similar_chunks

openai.api_key = os.getenv("OPENAI_API_KEY")

async def answer_question(query: str) -> str:
    indices = await search_similar_chunks(query)
    # Assuming you have a way to map indices back to document chunks
    context = " ".join([retrieve_chunk_by_index(idx) for idx in indices])
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:",
        max_tokens=150
    )
    
    return response.choices[0].text.strip()

def retrieve_chunk_by_index(index):
    # Implement this function to retrieve the chunk text by its index
    pass