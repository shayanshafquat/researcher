from typing import List, Dict, Any
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
from dataclasses import dataclass
import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# Load environment variables from .env.test
load_dotenv(".env.test")

@dataclass
class EvaluationResult:
    experiment_name: str
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: float

class RAGASEvaluator:
    def __init__(self, qa_dataset_path: str):
        """Initialize RAGAS evaluator with path to QA dataset"""
        self.qa_pairs = self._load_qa_dataset(qa_dataset_path)
        self._setup_evaluation_models()
        
    def _setup_evaluation_models(self):
        """Setup OpenAI LLM and embedding models for evaluation"""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Wrap models in RAGAS-compatible classes
        self.ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
        self.ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Initialize metrics with our models
        self.metrics = [
            LLMContextRecall(llm=self.ragas_llm), 
            FactualCorrectness(llm=self.ragas_llm), 
            Faithfulness(llm=self.ragas_llm),
            SemanticSimilarity(embeddings=self.ragas_embeddings)
        ]
        
    def _load_qa_dataset(self, path: str) -> List[Dict[str, Any]]:
        """Load QA pairs from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def evaluate_experiment(self, 
                          query_engine: Any, 
                          experiment_name: str) -> EvaluationResult:
        """Evaluate a single experiment using RAGAS metrics"""
        
        # Prepare data for RAGAS dataset
        evaluation_data = []
        
        for qa_pair in self.qa_pairs:
            question = qa_pair['question']
            response = query_engine.query(question)
            
            # Convert source nodes to text format
            contexts = [node.text for node in response.source_nodes]
            
            evaluation_data.append({
                "question": question,
                "answer": response.response,
                "contexts": contexts,
                "ground_truth": qa_pair.get('answer', '')  # Include ground truth if available
            })
        
        # Create RAGAS dataset
        dataset = Dataset.from_list(evaluation_data)
        
        # Run RAGAS evaluation
        evaluation = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )
        
        # Extract scores from evaluation results
        return EvaluationResult(
            experiment_name=experiment_name,
            faithfulness_score=evaluation['faithfulness'],
            answer_relevancy_score=evaluation['semantic_similarity'],
            context_precision_score=evaluation['factual_correctness'],
            context_recall_score=evaluation['context_recall']
        )
    
    def evaluate_all_experiments(self, 
                               experiments: Dict[str, Any]) -> pd.DataFrame:
        """Evaluate all experiments and return results as DataFrame"""
        results = []
        
        for name, query_engine in experiments.items():
            print(f"\nEvaluating: {name}")
            result = self.evaluate_experiment(query_engine, name)
            results.append(result)
            
        return pd.DataFrame([vars(r) for r in results]) 