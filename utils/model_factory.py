from abc import ABC, abstractmethod
import os
from openai import AsyncOpenAI
from huggingface_hub import InferenceClient
from config.model_config import ModelProvider, get_model_config
from typing import List, Dict
import json
import logging

logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        pass

    @abstractmethod
    async def generate_text_with_functions(
        self, 
        prompt: str, 
        system_prompt: str = None,
        functions: List[Dict] = None
    ) -> str:
        pass

class OpenAIModel(LLMInterface):
    def __init__(self):
        self.config = get_model_config().openai
        self.client = AsyncOpenAI(api_key=self.config.api_key)

    async def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise

    async def generate_text_with_functions(
        self, 
        prompt: str, 
        system_prompt: str = None,
        functions: List[Dict] = None
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=[{"type": "function", "function": f} for f in functions],
                tool_choice="auto"
            )
            
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                tool_call = tool_calls[0]
                return json.dumps({
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                })
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with functions: {str(e)}")
            raise

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

    async def generate_text_with_functions(
        self, 
        prompt: str, 
        system_prompt: str = None,
        functions: List[Dict] = None
    ) -> str:
        # Format the prompt to include function descriptions since local models
        # might not support native function calling
        formatted_prompt = f"""
        Available functions:
        {json.dumps(functions, indent=2)}
        
        Instructions: Based on the following prompt, choose the most appropriate function to call.
        Return your response in JSON format like: {{"name": "function_name", "arguments": {{"query": "search query"}}}}
        
        System: {system_prompt if system_prompt else ""}
        
        User: {prompt}
        """
        
        response = await self.generate_text(formatted_prompt)
        
        # Ensure response is valid JSON
        try:
            json.loads(response)
            return response
        except:
            # Fallback to a default function call if parsing fails
            return json.dumps({
                "name": "answer_from_document",
                "arguments": {"query": prompt}
            })

class ModelFactory:
    @staticmethod
    def get_model(provider: ModelProvider) -> LLMInterface:
        if provider == ModelProvider.OPENAI:
            return OpenAIModel()
        elif provider == ModelProvider.LOCAL:
            return LocalModel()
        else:
            raise ValueError(f"Unknown model provider: {provider}")