import openai
import os
from typing import List
from langchain.docstore.document import Document

openai.api_key = os.getenv("OPENAI_API_KEY")

async def generate_queries(num_queries: int = 5) -> List[str]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with generating diverse queries to summarize a scientific research paper."},
            {"role": "user", "content": f"""Generate {num_queries} diverse queries that would help in summarizing the key aspects of a scientific research paper. The queries should cover:

            1. The main research question or objective
            2. The methodology used
            3. Key findings and results
            4. Conclusions and implications
            5. Limitations and future research directions

            Please provide {num_queries} concise queries."""}
        ],
        max_tokens=200
    )
    queries = response.choices[0]['message']['content'].split('\n')
    return [query.strip() for query in queries if query.strip()]

async def summarize_document(chunks: List[Document]) -> str:
    context = " ".join([chunk.page_content for chunk in chunks])
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert research assistant capable of summarizing complex academic papers. Provide a comprehensive and insightful summary of the given research article."},
            {"role": "user", "content": f"""Please summarize the following research article in detail. Your summary should include:

            1. Title and authors (if available)
            2. Main research question or objective
            3. Key background information and context
            4. Methodology used in the study
            5. Primary findings and results
            6. Significant conclusions and their implications
            7. Any limitations mentioned in the study
            8. Potential future research directions suggested

            Organize the summary in a clear, coherent structure. Use academic language, but ensure it's accessible to a broader audience. If specific sections are unclear or missing, mention this in your summary.

            Here's the article text:

            {context}

            Please provide a detailed summary based on the above instructions."""}
        ],
        max_tokens=700  # Increased token limit for a more detailed summary
    )
    return response.choices[0]['message']['content']

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
