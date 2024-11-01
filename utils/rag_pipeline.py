from typing import List
from langchain.docstore.document import Document
from utils.model_factory import ModelFactory, ModelProvider

class RAGPipeline:
    def __init__(self, model_provider: ModelProvider):
        self.model = ModelFactory.get_model(model_provider)

    async def generate_queries(self, num_queries: int = 5) -> List[str]:
        system_prompt = "You are an AI assistant tasked with generating diverse queries to summarize a scientific research paper."
        prompt = f"""Generate {num_queries} diverse queries that would help in summarizing the key aspects of a scientific research paper. The queries should cover:

        1. The main research question or objective
        2. The methodology used
        3. Key findings and results
        4. Conclusions and implications
        5. Limitations and future research directions

        Please provide {num_queries} concise queries."""

        response = await self.model.generate_text(prompt, system_prompt)
        queries = response.split('\n')
        return [query.strip() for query in queries if query.strip()]

    async def summarize_document(self, chunks: List[Document]) -> str:
        context = " ".join([chunk.page_content for chunk in chunks])
        
        system_prompt = "You are an expert research assistant capable of summarizing complex academic papers."
        prompt = f"""Please summarize the following research article in detail. Your summary should include:

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

        Please provide a detailed summary based on the above instructions."""

        return await self.model.generate_text(prompt, system_prompt)

    async def answer_question(self, query: str, similar_chunks: List[Document]) -> str:
        context = " ".join([chunk.page_content for chunk in similar_chunks])
        
        system_prompt = "You are a helpful assistant. Answer the question based on the given context."
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        return await self.model.generate_text(prompt, system_prompt)
