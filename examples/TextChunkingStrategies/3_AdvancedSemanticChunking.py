import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # path to the file to chunk
FILE_ENCODING = "utf-8"              # file encoding
BREAKPOINT_THRESHOLD_TYPE = "percentile"   # "percentile", "standard_deviation", "interquartile", "gradient"
BREAKPOINT_THRESHOLD_AMOUNT = 95            # threshold value (meaning depends on type above)
# ──────────────────────────────────────────────────────────────────────────────

# Load a single text file
loader = TextLoader(str(DATA_FILE), encoding=FILE_ENCODING)
documents = loader.load()

# Split based on semantic similarity
semantic_splitter = SemanticChunker(
    AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    ),
    breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
    breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_AMOUNT,
)

semantic_chunks = semantic_splitter.split_documents(documents)
print(f"Semantic chunks: {len(semantic_chunks)}")
print(f"Sample semantic chunk content: {semantic_chunks[0].page_content[:200]}")