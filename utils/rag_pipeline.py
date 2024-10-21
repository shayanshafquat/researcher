import openai
import os
from typing import List
from langchain.docstore.document import Document

openai.api_key = os.getenv("OPENAI_API_KEY")

async def answer_question(query: str, similar_chunks: List[Document]) -> str:
    context = " ".join([chunk.page_content for chunk in similar_chunks])
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the given context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
        ],
        max_tokens=500
    )
    return response.choices[0]['message']['content']

def retrieve_chunk_by_index(index):
    # Implement this function to retrieve the chunk text by its index
    pass
