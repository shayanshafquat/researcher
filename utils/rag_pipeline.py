from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from utils.model_factory import ModelFactory, ModelProvider
from utils.search_utils import GoogleSearchTool, QueryAnalyzer, FunctionRegistry
import json
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RAGPipeline:
    def __init__(self, model_provider: ModelProvider):
        self.model = ModelFactory.get_model(model_provider)
        self.google_search = GoogleSearchTool()
        self.query_analyzer = QueryAnalyzer()
        self.function_registry = FunctionRegistry()
        logger.info(f"Initialized RAG Pipeline with model provider: {model_provider}")

    async def generate_queries(self, num_queries: int = 5) -> List[str]:
        logger.info(f"Generating {num_queries} queries for document summarization")
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
        logger.info(f"Summarizing document with {len(chunks)} chunks")
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
        logger.info(f"Processing question: {query}")
        context = " ".join([chunk.page_content for chunk in similar_chunks])
        
        # First, let the model decide which function to call
        system_prompt = """You are a helpful research assistant. Based on the question, decide whether to:
        1. Use the provided document context to answer (use answer_from_document)
        2. Search the web for recent or additional information (use google_search)
        Choose the appropriate function based on the nature of the question."""
        
        function_choice_prompt = f"""Question: {query}
        Document Context Preview: {context[:500]}..."""
        
        try:
            # Get function choice using function calling
            logger.info("Determining whether to use Google Search or document context")
            function_response = await self.model.generate_text_with_functions(
                function_choice_prompt,
                system_prompt,
                self.function_registry.get_function_definitions()
            )
            
            # Parse the function call
            function_call = json.loads(function_response)
            function_name = function_call.get("name")
            logger.info(f"Model chose to use function: {function_name}")
            
            if function_name == "google_search":
                logger.info("Performing Google search for additional information")
                search_results = await self.google_search.search(query)
                if search_results:
                    logger.info(f"Found {len(search_results)} relevant search results")
                    external_context = "\n".join([
                        f"Source ({result.title}): {result.content}"
                        for result in search_results
                    ])
                    
                    prompt = f"""Question: {query}
                    
                    Document Context: {context}
                    
                    Additional Information from Web Search:
                    {external_context}
                    
                    Please provide a comprehensive answer using both the document context and the external information. 
                    Clearly cite your sources when using external information."""
                    
                else:
                    logger.warning("No search results found, falling back to document context")
                    prompt = f"Question: {query}\n\nDocument Context: {context}"
            else:
                logger.info("Using document context for answering")
                prompt = f"Question: {query}\n\nDocument Context: {context}"
            
            # Generate final answer
            system_prompt = """You are a helpful research assistant. Provide a clear and accurate answer based on the available information.
            When using external sources, clearly indicate this with proper citations."""
            
            answer = await self.model.generate_text(prompt, system_prompt)
            
            # Add source attribution if external search was used
            if function_name == "google_search" and search_results:
                logger.info("Adding source citations to the answer")
                answer += "\n\nSources:\n"
                for result in search_results:
                    answer += f"- {result.title}: {result.url}\n"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            logger.info("Falling back to document context due to error")
            # Fallback to document context if function calling fails
            prompt = f"Question: {query}\n\nDocument Context: {context}"
            return await self.model.generate_text(prompt, system_prompt)
