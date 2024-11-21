import logging
from pathlib import Path
import asyncio
import argparse
from researcher.testing.data_preparation import TestDataPreparer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def regenerate_qa_pairs(preparer: TestDataPreparer):
    """Regenerate QA pairs for existing papers"""
    # Clear existing QA pairs metadata
    preparer.qa_pairs_metadata = {}
    preparer._save_cache(preparer.qa_pairs_metadata, preparer.qa_cache)
    
    # Clear existing QA files
    for qa_file in preparer.qa_pairs_dir.glob("*_qa.json"):
        qa_file.unlink()
        
    logger.info("Cleared existing QA pairs. Regenerating...")
    
    # Generate new QA pairs
    new_qa_pairs = await preparer.generate_qa_pairs()
    logger.info(f"Regenerated QA pairs for {len(new_qa_pairs)} papers")
    
    return new_qa_pairs

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare test data and generate QA pairs')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate QA pairs for existing papers')
    args = parser.parse_args()

    # Initialize data directory
    data_dir = Path("tests/test_data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize data preparer
    preparer = TestDataPreparer(data_dir)
    
    if args.regenerate:
        # Regenerate QA pairs for existing papers
        logger.info("Regenerating QA pairs for existing papers...")
        new_qa_pairs = await regenerate_qa_pairs(preparer)
    else:
        # Normal flow: download papers and generate QA pairs
        logger.info("Downloading research papers...")
        downloaded_papers = preparer.download_papers(max_papers=15)
        logger.info(f"Downloaded/loaded {len(downloaded_papers)} papers")
        
        logger.info("Generating QA pairs...")
        new_qa_pairs = await preparer.generate_qa_pairs()
        logger.info(f"Generated QA pairs for {len(new_qa_pairs)} new papers")
    
    # Load all QA pairs
    all_qa_pairs = await preparer.get_all_qa_pairs()
    logger.info(f"Total papers with QA pairs: {len(all_qa_pairs)}")

if __name__ == "__main__":
    asyncio.run(main()) 