from pydantic import BaseModel, ConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class GoogleSearchConfig(BaseModel):
    """Configuration for Google Custom Search API"""
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    search_engine_id: str = os.getenv("GOOGLE_CSE_ID", "")
    max_results: int = 5
    
    model_config = ConfigDict(protected_namespaces=())

SEARCH_CONFIG = GoogleSearchConfig() 