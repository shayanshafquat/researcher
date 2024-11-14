from abc import ABC, abstractmethod
import os
from openai import AsyncOpenAI
from huggingface_hub import InferenceClient
from config.model_config import ModelProvider, get_model_config
from typing import List, Dict
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            logger.info("Sending request to OpenAI for function selection")
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=[{"type": "function", "function": f} for f in functions],
                tool_choice="auto",
                temperature=0.1
            )
            
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                tool_call = tool_calls[0]
                logger.info(f"Selected function: {tool_call.function.name}")
                return json.dumps({
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                })
            
            logger.warning("No function selected, falling back to document context")
            return json.dumps({
                "name": "answer_from_document",
                "arguments": {"query": prompt}
            })

        except Exception as e:
            logger.error(f"Error in function calling with OpenAI: {str(e)}")
            logger.info("Falling back to document context")
            return json.dumps({
                "name": "answer_from_document",
                "arguments": {"query": prompt}
            })

class LocalModel(LLMInterface):
    def __init__(self):
        self.config = get_model_config().local
        self.client = InferenceClient(token=self.config.hf_api_key)
        
    async def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })

            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating text with local model: {str(e)}")
            raise

    async def generate_text_with_functions(
        self, 
        prompt: str, 
        system_prompt: str = None,
        functions: List[Dict] = None
    ) -> str:
        try:
            logger.info(f"Generating text with functions using {self.config.model_id}")
            # Format the system message to include function descriptions
            function_descriptions = "\n".join([
                f"Function: {f['name']}\n"
                f"Description: {f['description']}\n"
                f"Parameters: {json.dumps(f['parameters'], indent=2)}\n"
                for f in functions
            ])

            formatted_system_prompt = f"""You are a helpful assistant that chooses between available functions.
            
Available Functions:
{function_descriptions}

Instructions:
1. Analyze the user's query and context
2. Choose the most appropriate function based on the query
3. Return ONLY a JSON response in the following format:
{{"name": "function_name", "arguments": {{"query": "your query"}}}}

Additional Context: {system_prompt if system_prompt else 'No additional context provided.'}"""

            messages = [
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": prompt}
            ]

            logger.info("Sending request to model for function selection")
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=0.2,
                max_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p
            )

            response_text = response.choices[0].message.content.strip()
            logger.info(f"Received response from model: {response_text[:100]}...")

            # Clean up the response
            try:
                if "```json" in response_text:
                    logger.info("Cleaning up markdown JSON formatting")
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].strip()
                
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    response_text = response_text[json_start:json_end]

                parsed_response = json.loads(response_text)
                
                if all(key in parsed_response for key in ["name", "arguments"]):
                    if parsed_response["name"] in ["google_search", "answer_from_document"]:
                        logger.info(f"Successfully parsed function call: {parsed_response['name']}")
                        return json.dumps(parsed_response)
                
                logger.warning(f"Invalid function call format: {response_text}")
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing model response: {str(e)}")
                logger.warning(f"Raw response: {response_text}")

            logger.info("Falling back to document context")
            return json.dumps({
                "name": "answer_from_document",
                "arguments": {"query": prompt}
            })

        except Exception as e:
            logger.error(f"Error in function calling: {str(e)}")
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