from typing import List, Dict, Optional
import json
from pathlib import Path
import logging
import asyncio
from ..core.utils.model_factory import LocalModel
from ..core.config.model_config import get_model_config

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    def __init__(self):
        self.model = LocalModel()
        self.config = get_model_config().local
        logger.info(f"Initialized LocalModel with {self.config.model_id}")
        
        # Define different question types for diverse Q&A generation
        self.question_strategies = [
            {
                "type": "factual",
                "instruction": "Generate a factual question about specific details, methods, or findings."
            },
            {
                "type": "conceptual",
                "instruction": "Create a question about the main concepts or theories discussed."
            },
            {
                "type": "comparative",
                "instruction": "Form a question that requires comparing different aspects mentioned."
            },
            {
                "type": "analytical",
                "instruction": "Generate a question about analyzing implications or significance."
            },
            {
                "type": "methodology",
                "instruction": "Ask about specific methods or experimental procedures."
            }
        ]
        
    async def generate_qa_pair(self, context: str, strategy: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Generate a single question-answer pair using a specific strategy"""
        # Simplified system prompt
        system_prompt = "You are a research paper expert that generates precise question-answer pairs in JSON format."
        
        # More structured prompt template
        prompt_template = """Given this research paper excerpt, create a {type} question and answer pair.

Context (excerpt from research paper):
---
{context}
---

Instructions:
1. Generate ONE {type} question based on the context
2. Provide a detailed answer using information from the context
3. Return ONLY a JSON object in this EXACT format:
{{
    "question": "Write your question here",
    "answer": "Write your detailed answer here",
    "type": "{type}"
}}

Remember:
- Question should require understanding the context
- Answer should be comprehensive and accurate
- Response must be valid JSON only"""
        
        try:
            # Split context into smaller chunks and select a relevant portion
            chunk_size = 2000  # Reduced chunk size
            chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]
            selected_chunk = chunks[0].strip()
            
            # Generate response
            response_text = await self.model.generate_text(
                prompt=prompt_template.format(
                    context=selected_chunk,
                    type=strategy['type']
                ),
                system_prompt=system_prompt
            )
            
            # Clean and parse the response
            try:
                # Remove any markdown formatting and extra whitespace
                response_text = response_text.strip()
                
                # Extract JSON part
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    
                    # Log the extracted JSON string
                    logger.debug(f"Attempting to parse JSON: {json_str}")
                    
                    try:
                        qa_pair = json.loads(json_str)
                        
                        # Validate the structure and content
                        required_keys = ["question", "answer", "type"]
                        if all(key in qa_pair for key in required_keys):
                            if all(isinstance(qa_pair[key], str) and qa_pair[key].strip() for key in required_keys):
                                logger.info(f"Successfully generated {qa_pair['type']} question")
                                return qa_pair
                            
                        logger.warning("Invalid QA pair structure")
                        return None
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error: {str(e)}")
                        logger.debug(f"Failed JSON string: {json_str}")
                        return None
                else:
                    logger.warning("No JSON object found in response")
                    logger.debug(f"Full response: {response_text}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                logger.debug(f"Raw response: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error in QA generation: {str(e)}")
            return None
    
    async def generate_qa_pairs(self, context: str, num_pairs: int = 5) -> List[Dict[str, str]]:
        """Generate multiple question-answer pairs using different strategies"""
        qa_pairs = []
        
        # Ensure we use different strategies for each question
        strategies = []
        while len(strategies) < num_pairs:
            strategies.extend(self.question_strategies)
        strategies = strategies[:num_pairs]
        
        # Generate pairs sequentially instead of concurrently
        for strategy in strategies:
            try:
                qa_pair = await self.generate_qa_pair(context, strategy)
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                logger.error(f"Error generating QA pair with strategy {strategy['type']}: {str(e)}")
                continue
        
        logger.info(f"Successfully generated {len(qa_pairs)} diverse QA pairs out of {num_pairs} attempts")
        return qa_pairs

    def save_synthetic_dataset(self, qa_pairs: List[Dict[str, str]], output_path: Path):
        """Save generated QA pairs to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(qa_pairs, f, indent=2)

    @staticmethod
    async def generate_dataset(context: str, num_pairs: int = 5, output_path: Path = None) -> List[Dict[str, str]]:
        """Static method to generate and optionally save a dataset"""
        generator = SyntheticDataGenerator()
        qa_pairs = await generator.generate_qa_pairs(context, num_pairs)
        
        if output_path:
            generator.save_synthetic_dataset(qa_pairs, output_path)
            
        return qa_pairs