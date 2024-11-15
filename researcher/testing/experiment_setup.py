from typing import Dict, Any, List
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostProcessor
import logging

logger = logging.getLogger(__name__)

class RAGExperimentManager:
    def __init__(self, documents: List[Document], llm: Any):
        self.documents = documents
        self.llm = llm
        self.service_context = ServiceContext.from_defaults(llm=self.llm)
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        
    def setup_experiments(self) -> Dict[str, RetrieverQueryEngine]:
        """Configure different RAG setups for evaluation"""
        experiments = {}
        
        # Basic retriever
        basic_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=2
        )
        experiments["basic"] = RetrieverQueryEngine(basic_retriever)
        
        # MMR retriever
        mmr_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=4
        )
        mmr_engine = RetrieverQueryEngine(
            mmr_retriever,
            node_postprocessors=[SimilarityPostProcessor(similarity_cutoff=0.7)]
        )
        experiments["mmr"] = mmr_engine
        
        # Add more experimental setups here...
        
        return experiments
    
    def run_experiment(
        self,
        experiment_name: str,
        query: str,
        query_engine: RetrieverQueryEngine
    ) -> Dict[str, Any]:
        """Run a single experiment and return results"""
        try:
            response = query_engine.query(query)
            return {
                "experiment": experiment_name,
                "query": query,
                "response": response,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in experiment {experiment_name}: {str(e)}")
            return {
                "experiment": experiment_name,
                "query": query,
                "error": str(e),
                "success": False
            } 