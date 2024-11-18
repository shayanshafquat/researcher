import pytest
import os
import logging
from pathlib import Path
import pymongo
from typing import Generator
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(".env.test")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def test_db() -> Generator:
    """Create a test MongoDB instance for storing test results"""
    client = pymongo.MongoClient(os.getenv("TEST_MONGODB_URI", "mongodb://localhost:27017/"))
    db = client.test_rag_db
    yield db
    client.drop_database('test_rag_db')
    client.close()

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create and return a temporary directory for test data"""
    data_dir = Path("tests/test_data")
    data_dir.mkdir(exist_ok=True)
    return data_dir