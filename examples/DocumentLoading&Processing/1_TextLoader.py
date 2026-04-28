from pathlib import Path
from langchain_community.document_loaders import TextLoader

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # path to the text file to load
FILE_ENCODING = "utf-8"                                      # file encoding
CONTENT_PREVIEW_LENGTH = 200                                 # characters to show in preview
# ──────────────────────────────────────────────────────────────────────────────

# Load a single text file
loader = TextLoader(str(DATA_FILE), encoding=FILE_ENCODING)
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Content preview: {documents[0].page_content[:CONTENT_PREVIEW_LENGTH]}")
print(f"Metadata: {documents[0].metadata}")
