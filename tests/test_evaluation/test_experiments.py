import pytest
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from tests.config.test_config import ExperimentalSetup
from tests.evaluation.ragas_evaluator import RAGASEvaluator
import json
import os
from llama_index.readers.file import PDFReader

@pytest.fixture
def test_documents():
    """Load a single PDF document from test_data directory"""
    # Specify the exact PDF file path
    pdf_path = os.path.join("test_data", "papers", "1302.3560v1.pdf")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    # Load PDF using PDFReader
    reader = PDFReader()
    documents = reader.load_data(pdf_path)
    
    print(f"Loaded {len(documents)} pages from PDF")
    
    # Add metadata to documents
    for doc in documents:
        doc.metadata.update({
            "file_name": "1302.3560v1.pdf",
            "source": pdf_path
        })
    
    return documents

@pytest.fixture
def test_index(test_documents):
    """Create vector store index from test documents"""
    print(f"Creating index from {len(test_documents)} documents")
    return VectorStoreIndex.from_documents(
        documents=test_documents,
        show_progress=True
    )

def test_experimental_evaluation(test_index):
    # Setup experiments
    setup = ExperimentalSetup()
    experiments = setup.get_experiments(test_index)
    
    # Initialize RAGAS evaluator with QA pairs
    # qa_path = os.path.join("./..", "test_data", "qa_pairs", "1302.3560v1_qa.json")
    qa_path = "./test_data/qa_pairs/1302.3560v1_qa.json"
    evaluator = RAGASEvaluator(qa_path)
    
    # Run evaluation
    results_df = evaluator.evaluate_all_experiments(experiments)
    
    # Print results for debugging
    print("\nEvaluation Results:")
    print(results_df)
    
    # Basic validation
    assert not results_df.empty
    assert all(all(score >= 0 for score in scores) for scores in results_df['faithfulness_score'])
    
    # Save results
    results_df.to_csv('test_evaluation_results.csv', index=False) 