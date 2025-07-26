import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API Key - check both possible environment variable names
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Debug: Check if API key is loaded
if not GEMINI_API_KEY:
    print("WARNING: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables")
else:
    print(f"API Key loaded: {GEMINI_API_KEY[:10]}...")

# Vector DB Directory
VECTOR_DB_DIR = "./db"

# PDF Path
PDF_PATH = "./data/hsc_bangla_1st.pdf"

# Chunking config
CHUNK_SIZE = 300

# Chroma Collection Name
COLLECTION_NAME = "hsc_bangla"