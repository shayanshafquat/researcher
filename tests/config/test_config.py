from typing import Dict, Any
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from researcher.core.config.model_config import ModelConfig, ModelProvider

class ExperimentalSetup:
    def __init__(self):
        configure_test_settings()
        
    def setup_naive_rag(self, index: VectorStoreIndex) -> BaseQueryEngine:
        return index.as_query_engine(
            similarity_top_k=3,  # Reduced for more focused retrieval
            response_mode="compact"
        )
    
    def setup_llm_rerank(self, index: VectorStoreIndex) -> BaseQueryEngine:
        retriever = index.as_retriever(similarity_top_k=5)
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-12-v2", 
            top_n=2
        )
        return RetrieverQueryEngine(
            retriever=retriever, 
            node_postprocessors=[rerank],
            response_mode="compact"
        )

    def setup_hyde(self, index: VectorStoreIndex) -> BaseQueryEngine:
        hyde_query_transform = HyDEQueryTransform(
            include_original=True
        )
        retriever = index.as_retriever(
            similarity_top_k=5,
            query_transform=hyde_query_transform
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            response_mode="compact"
        )
    
    def setup_hyde_with_rerank(self, index: VectorStoreIndex) -> BaseQueryEngine:
        hyde_query_transform = HyDEQueryTransform(
            include_original=True
        )
        retriever = index.as_retriever(
            similarity_top_k=5,
            query_transform=hyde_query_transform
        )
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-12-v2", 
            top_n=2
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[rerank],
            response_mode="compact"
        )

    def setup_mmr(self, index: VectorStoreIndex) -> BaseQueryEngine:
        retriever = index.as_retriever(
            similarity_top_k=5,
            mmr_threshold=0.7,
        )
        return RetrieverQueryEngine(retriever=retriever)

    def setup_multi_query(self, index: VectorStoreIndex) -> BaseQueryEngine:
        retriever = MultiStepQueryEngine.from_defaults(
            query_engine=index.as_query_engine(),
            similarity_top_k=10,
            num_queries=3,
        )
        return RetrieverQueryEngine(retriever=retriever)

    def setup_sentence_window(self, index: VectorStoreIndex) -> BaseQueryEngine:
        return index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[
                MetadataReplacementPostProcessor(
                    target_metadata_key="window",
                    replace_metadata_key="context"
                )
            ]
        )

    def setup_sentence_window_with_rerank(self, index: VectorStoreIndex) -> BaseQueryEngine:
        retriever = index.as_retriever(similarity_top_k=10)
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-12-v2", 
            top_n=3
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[
                MetadataReplacementPostProcessor(
                    target_metadata_key="window",
                    replace_metadata_key="context"
                ),
                rerank
            ]
        )

    def get_experiments(self, index: VectorStoreIndex) -> Dict[str, BaseQueryEngine]:
        return {
            "Classic VDB + Naive RAG": self.setup_naive_rag(index)
            # "Classic VDB + LLM Rerank": self.setup_llm_rerank(index),
            # "Classic VDB + HyDE": self.setup_hyde(index)
            # "Classic VDB + HyDE + LLM Rerank": self.setup_hyde_with_rerank(index),
            # "Classic VDB + MMR": self.setup_mmr(index)
            # "Classic VDB + Multi Query": self.setup_multi_query(index),
            # "Sentence window retrieval": self.setup_sentence_window(index),
            # "Sentence window + LLM Rerank": self.setup_sentence_window_with_rerank(index),
        } 

def configure_test_settings():
    """Configure global settings for tests"""
    model_config = ModelConfig()
    model_config.active_provider = ModelProvider.OPENAI
    openai_config = model_config.get_active_config()
    
    # Initialize OpenAI LLM
    llm = OpenAI(
        model=openai_config.model_name,
        api_key=openai_config.api_key,
        temperature=0.1,  # Reduced for more consistent responses
        max_tokens=openai_config.max_tokens,
    )
    
    # Initialize OpenAI embeddings
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key=openai_config.api_key,
        embed_batch_size=100,
    )
    
    # Configure settings using the new approach
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512  # Reduced for better context
    Settings.chunk_overlap = 50
    Settings.num_output = openai_config.max_tokens