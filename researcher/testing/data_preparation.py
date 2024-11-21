import arxiv
import os
from pathlib import Path
import logging
from tqdm import tqdm
import json
from typing import List, Dict
import hashlib
from .synthetic_data import SyntheticDataGenerator
import PyPDF2
import asyncio

logger = logging.getLogger(__name__)

class TestDataPreparer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.papers_dir = data_dir / "papers"
        self.qa_pairs_dir = data_dir / "qa_pairs"
        self.papers_dir.mkdir(exist_ok=True)
        self.qa_pairs_dir.mkdir(exist_ok=True)
        
        # Cache file paths
        self.papers_cache = self.data_dir / "papers_cache.json"
        self.qa_cache = self.data_dir / "qa_cache.json"
        
        # Initialize caches
        self.papers_metadata = self._load_cache(self.papers_cache)
        self.qa_pairs_metadata = self._load_cache(self.qa_cache)
        
        self.search_queries = [
            "LLM cognitive science decision making",
            "transformer models cognitive psychology",
            "neural language models decision theory",
            "cognitive architecture large language models",
            "human decision making AI comparison"
        ]
        
        # Initialize synthetic data generator
        self.generator = SyntheticDataGenerator()

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache from file if it exists"""
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache_data: Dict, cache_file: Path):
        """Save cache to file"""
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def download_papers(self, max_papers: int = 15) -> List[Dict]:
        """Download papers if not already cached"""
        if len(self.papers_metadata) >= max_papers:
            logger.info(f"Using {max_papers} cached papers")
            return list(self.papers_metadata.values())[:max_papers]

        client = arxiv.Client()
        downloaded_papers = []

        for query in self.search_queries:
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )

            for paper in tqdm(client.results(search), desc=f"Downloading papers for: {query}"):
                paper_id = paper.get_short_id()
                # Fix: Create proper file path
                pdf_path = self.papers_dir / f"{paper_id}.pdf"
                
                if paper_id not in self.papers_metadata:
                    try:
                        # Fix: Create directory if it doesn't exist
                        self.papers_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Fix: Download with correct path handling
                        paper.download_pdf(filename=str(pdf_path))
                        
                        paper_info = {
                            "id": paper_id,
                            "title": paper.title,
                            "abstract": paper.summary,
                            "pdf_path": str(pdf_path)
                        }
                        logger.info(f"Downloaded paper: {paper_info['title']}")
                        self.papers_metadata[paper_id] = paper_info
                        downloaded_papers.append(paper_info)
                        
                        if len(downloaded_papers) + len(self.papers_metadata) >= max_papers:
                            break
                    except Exception as e:
                        logger.error(f"Error downloading paper {paper_id}: {str(e)}")
                        # Fix: Clean up partial downloads
                        if pdf_path.exists():
                            pdf_path.unlink()
                        continue

            if len(downloaded_papers) + len(self.papers_metadata) >= max_papers:
                break

        self._save_cache(self.papers_metadata, self.papers_cache)
        return downloaded_papers

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return text

    async def generate_qa_pairs(self) -> Dict[str, List[Dict[str, str]]]:
        """Generate QA pairs for papers if not already cached"""
        new_qa_pairs = {}

        for paper_id, paper_info in tqdm(self.papers_metadata.items(), desc="Generating QA pairs"):
            if paper_id not in self.qa_pairs_metadata:
                try:
                    text = self.extract_text_from_pdf(paper_info['pdf_path'])
                    qa_pairs = await self.generator.generate_qa_pairs(text)
                    
                    if qa_pairs:  # Only save if we got valid QA pairs
                        qa_file = self.qa_pairs_dir / f"{paper_id}_qa.json"
                        with open(qa_file, 'w') as f:
                            json.dump(qa_pairs, f, indent=2)
                        
                        self.qa_pairs_metadata[paper_id] = {
                            "qa_file": str(qa_file),
                            "num_pairs": len(qa_pairs)
                        }
                        new_qa_pairs[paper_id] = qa_pairs
                        logger.info(f"Generated {len(qa_pairs)} QA pairs for paper {paper_id}")
                except Exception as e:
                    logger.error(f"Error generating QA pairs for {paper_id}: {str(e)}")
                    continue

        self._save_cache(self.qa_pairs_metadata, self.qa_cache)
        return new_qa_pairs

    async def get_all_qa_pairs(self) -> Dict[str, List[Dict[str, str]]]:
        """Load all cached QA pairs"""
        all_qa_pairs = {}
        for paper_id, qa_info in self.qa_pairs_metadata.items():
            try:
                with open(qa_info['qa_file'], 'r') as f:
                    all_qa_pairs[paper_id] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading QA pairs for {paper_id}: {str(e)}")
                continue
        return all_qa_pairs 