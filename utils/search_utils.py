from typing import List, Dict, Any, Optional
import aiohttp
import logging
from config.search_config import SEARCH_CONFIG
import json

logger = logging.getLogger(__name__)

class SearchResult:
    def __init__(self, title: str, link: str, snippet: str):
        self.title = title
        self.link = link
        self.snippet = snippet
        
    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet
        }

class GoogleSearchTool:
    def __init__(self):
        self.api_key = SEARCH_CONFIG.api_key
        self.search_engine_id = SEARCH_CONFIG.search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    async def search(self, query: str) -> List[SearchResult]:
        """Perform Google Custom Search"""
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': SEARCH_CONFIG.max_results
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            SearchResult(
                                title=item.get('title', ''),
                                link=item.get('link', ''),
                                snippet=item.get('snippet', '')
                            )
                            for item in data.get('items', [])
                        ]
                    else:
                        logger.error(f"Google Search API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error performing Google search: {str(e)}")
            return []

class FunctionRegistry:
    @staticmethod
    def get_function_definitions() -> List[Dict[str, Any]]:
        """Get the function definitions for the LLM"""
        return [
            {
                "name": "google_search",
                "description": "Search the web for recent or additional information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "answer_from_document",
                "description": "Answer questions using the provided document context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to answer from the document"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

class QueryAnalyzer:
    def __init__(self):
        self.temporal_keywords = [
            'latest', 'recent', 'current', 'new', 'update',
            'today', 'now', 'modern', 'upcoming', 'trend'
        ]
        self.external_keywords = [
            'compare', 'other', 'alternative', 'different',
            'outside', 'beyond', 'additional', 'more'
        ]

    def analyze_query(self, query: str, document_context: str) -> Dict[str, Any]:
        """
        Analyze the query to determine if it needs external search
        """
        needs_external = False
        reasons = []

        # Check for temporal keywords
        if any(keyword in query.lower() for keyword in self.temporal_keywords):
            needs_external = True
            reasons.append("Query contains temporal keywords")

        # Check for comparison/external reference keywords
        if any(keyword in query.lower() for keyword in self.external_keywords):
            needs_external = True
            reasons.append("Query suggests need for external information")

        # Check if the query topic is covered in the document context
        # This is a simple check; you might want to use more sophisticated methods
        main_terms = [word.lower() for word in query.split() if len(word) > 3]
        if not any(term in document_context.lower() for term in main_terms):
            needs_external = True
            reasons.append("Query topic not found in document context")

        return {
            "needs_external_search": needs_external,
            "reasons": reasons
        } 