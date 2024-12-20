---
title: "ResearchGPT: AI Research Assistant"
emoji: "🚀"
colorFrom: "blue"
colorTo: "purple"
sdk: "streamlit"
sdk_version: "1.39.0"
app_file: "app.py"
pinned: false
license: "mit"
---

# ResearchGPT: AI Research Assistant

This application is a Retrieval-Augmented Generation (RAG) based system that allows users to upload documents or provide links to documents (including arXiv papers), and then ask questions about the content of those documents. It supports both OpenAI API models and open-source LLMs like Mistral-7B-Instruct, giving users flexibility in choosing their preferred model.

## Features

- Upload PDF documents
- Process document links, including arXiv papers
- Summarize uploaded documents
- Ask questions about uploaded documents
- Flexible Model Selection:
  - OpenAI GPT Models: High-performance option using OpenAI's API
  - Open Source LLMs: Cost-effective alternative using Mistral-7B-Instruct
- Dynamic model switching without losing context
- Cached responses for better performance
- Google Search Integration: Both models can perform web searches for recent or external information

## Tech Stack

### Core Technologies
- FastAPI: Backend API framework
- Streamlit: Frontend user interface
- LlamaIndex: Core RAG implementation and query engines
- Langchain: Document processing and model integrations
- FAISS: Vector database for efficient similarity search
- OpenAI API: For text embeddings and question answering
- Hugging Face Hub: For accessing open-source LLMs
- Poetry: For dependency management

### Evaluation & Testing
- RAGAS: For automated RAG pipeline evaluation
- Pytest: For testing framework
- GitHub Actions: For CI/CD and automated evaluation

### Document Processing
- PyMuPDF: For PDF processing
- Sentence Transformers: For text embeddings and reranking
- NLTK: For text processing and tokenization

## Model Support

### OpenAI Models
- Default model: GPT-3.5-turbo
- Suitable for: Production environments requiring high accuracy
- Requires: OpenAI API key

### Open Source Models
- Default model: Mistral-7B-Instruct-v0.3
- Suitable for: Development, testing, or cost-sensitive deployments
- Requires: Hugging Face API key
- Advantages: No usage costs, full control over the model

## RAG Pipeline Evaluation

### Evaluation Infrastructure
- Automated evaluation pipeline using GitHub Actions
- RAGAS metrics for comprehensive assessment
- Continuous evaluation on pull requests
- Detailed performance tracking and reporting

### Evaluation Metrics

1. **Faithfulness Score**: Measures how accurately the generated answers reflect the source content
2. **Answer Relevancy**: Evaluates the semantic relevance of answers to questions
3. **Context Precision**: Assesses the accuracy of retrieved context
4. **Context Recall**: Measures the completeness of retrieved relevant information

### Experimental Results

Current evaluation results for different RAG configurations:

| Experiment | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|------------|--------------|------------------|-------------------|----------------|
| Classic VDB + Naive RAG | [1.0, 1.0, 1.0, 1.0] | [0.9806, 0.9933, 0.9985, 0.9718] | [1.0, 0.83, 1.0, 0.8] | [1.0, 1.0, 0.6666, 1.0] |
| Classic VDB + LLM Rerank | [1.0, 1.0, 1.0, 1.0] | [0.9806, 0.9901, 0.9933, 0.9713] | [1.0, 0.67, 0.67, 0.8] | [1.0, 1.0, 0.6666, 1.0] |

### RAG Configurations Tested

1. **Classic VDB + Naive RAG**
   - Basic vector database retrieval with top-k=3
   - Direct question answering with compact response mode
   - Optimized for speed and simplicity
   - Shows consistently high faithfulness and answer relevancy

2. **Classic VDB + LLM Rerank**
   - Enhanced retrieval with top-k=5
   - Uses cross-encoder model (ms-marco-MiniLM-L-12-v2) for reranking
   - Improved precision through semantic reranking
   - Maintains high faithfulness while slightly trading off context precision

2. **MMR (Maximal Marginal Relevance)**
   - Balances relevance and diversity
   - Configurable with mmr_threshold=0.7

3. **Advanced Combinations**
   - Sentence Window Retrieval
   - Multi-Query Expansion
   - HyDE + Rerank
   - Window + HyDE

### Continuous Evaluation Pipeline

Our GitHub Actions workflow automatically:
- Runs on pull requests to main branch
- Executes comprehensive RAG evaluations
- Generates CSV reports with detailed metrics
- Posts results as PR comments
- Archives evaluation artifacts

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-based-qa-app.git
   cd rag-based-qa-app
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here  # Required for OpenAI models
   HF_API_KEY=your_huggingface_api_key_here # Required for Mistral and other open-source models
   GOOGLE_API_KEY=your_google_api_key_here  # Required for Google Search
   GOOGLE_CSE_ID=your_google_cse_id_here    # Required for Google Search
   ```

4. Run the application:
   
   Option 1: Run both services together:
   ```bash
   poetry run start
   ```
   
   Option 2: Run services separately:
   ```bash
   # Terminal 1 - Backend
   poetry run python main.py
   
   # Terminal 2 - Frontend
   poetry run streamlit run app.py
   ```

## Usage

1. Upload a document or paste a document link (including arXiv links).
2. Select your preferred model provider (OpenAI or Local LLM) from the sidebar.
3. Wait for the document to be processed.
4. Click on "Summarize Document" to get an overview of the uploaded document.
5. Ask questions about the document in the provided text input.
6. View the AI-generated answers based on the document's content.

## Google Search Integration

Both OpenAI and Mistral models can perform Google searches to find recent or external information when needed. This feature enhances the model's ability to provide up-to-date and comprehensive answers.

![Google Search Integration](images/3.png)

## Model Selection Guide

### When to Use OpenAI Models:
- Need highest accuracy and performance
- Working with complex academic papers
- Require production-grade responses
- Budget allows for API usage

### When to Use Local LLMs (Mistral):
- Development and testing
- Cost-sensitive operations
- Privacy concerns with external APIs
- Need for offline capabilities
- Sufficient for basic summarization and Q&A

## Application Interface

### Document Upload and Summary
![Document Upload and Summary](images/1.png)

### Q&A Interface
![Q&A Interface](images/2.png)

## Project Structure

```
researcher/
├── core/
│   ├── config/
│   │   └── model_config.py    # Model configurations
│   ├── routers/
│   │   └── document_routes.py # API endpoints
│   └── utils/
│       ├── vector_store.py    # FAISS operations
│       ├── rag_pipeline.py    # RAG implementation
│       └── text_processing.py # Text processing
├── testing/
│   └── data_preparation.py    # Test data generation
└── tests/
    ├── evaluation/
    │   └── ragas_evaluator.py # RAGAS evaluation
    ├── config/
    │   └── test_config.py     # Test configurations
    └── test_evaluation/
        └── test_experiments.py # Evaluation tests
```

## Running with Docker

1. Build the Docker image:
   ```
   docker build -t rag-qa-app .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 -p 8501:8501 --env-file .env rag-qa-app
   ```

3. Access the application:
   - FastAPI backend: `http://localhost:8000`
   - Streamlit frontend: `http://localhost:8501`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of particular interest include:
- Adding support for additional open-source models
- Improving model response caching
- Enhancing the RAG pipeline
- UI/UX improvements
- Adding new RAG configurations for evaluation
- Improving evaluation metrics and benchmarks
- Enhancing the RAGAS evaluation pipeline
- Optimizing retrieval strategies based on evaluation results

## License

This project is licensed under the MIT License.
