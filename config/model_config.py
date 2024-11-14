from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv()

class ModelProvider(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"

class LocalModelConfig(BaseModel):
    """Configuration for Hugging Face Inference API"""
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_api_key: str = os.getenv("HF_API_KEY", "")
    max_new_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

    model_config = ConfigDict(protected_namespaces=())

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API"""
    model_name: str = "gpt-3.5-turbo"
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    max_tokens: int = 500
    temperature: float = 0.7

    model_config = ConfigDict(protected_namespaces=())

class ModelConfig(BaseModel):
    """Main configuration class for model settings"""
    openai: OpenAIConfig = OpenAIConfig()
    local: LocalModelConfig = LocalModelConfig()
    active_provider: ModelProvider = ModelProvider.OPENAI

    model_config = ConfigDict(protected_namespaces=())

    def get_active_config(self):
        """Returns the configuration for the currently active provider"""
        if self.active_provider == ModelProvider.OPENAI:
            return self.openai
        return self.local

MODEL_CONFIG = ModelConfig()
def update_model_config(provider: ModelProvider):
    """Updates the active model provider"""
    MODEL_CONFIG.active_provider = provider

def get_model_config() -> ModelConfig:
    """Returns the current model configuration"""
    return MODEL_CONFIG
