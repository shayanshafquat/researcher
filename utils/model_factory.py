from abc import ABC, abstractmethod
import os
import openai
from huggingface_hub import InferenceClient
from config.model_config import ModelProvider, get_model_config

class LLMInterface(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        pass

class OpenAIModel(LLMInterface):
    def __init__(self):
        self.config = get_model_config().openai
        openai.api_key = self.config.api_key

    async def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response.choices[0]['message']['content']

class LocalModel(LLMInterface):
    def __init__(self):
        self.config = get_model_config().local
        self.client = InferenceClient(token=self.config.hf_api_key)

    async def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        stream = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            stream=False
        )
        return stream.choices[0].message.content

class ModelFactory:
    @staticmethod
    def get_model(provider: ModelProvider) -> LLMInterface:
        if provider == ModelProvider.OPENAI:
            return OpenAIModel()
        elif provider == ModelProvider.LOCAL:
            return LocalModel()
        else:
            raise ValueError(f"Unknown model provider: {provider}")