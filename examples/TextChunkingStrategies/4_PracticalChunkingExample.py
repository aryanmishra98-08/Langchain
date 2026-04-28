from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # path to the file to chunk
CHUNK_SIZE = 1000                              # max characters per chunk
CHUNK_OVERLAP = 200                            # overlap between consecutive chunks
SEPARATORS = ["\n\n", "\n", " ", ""]          # ordered list of split separators
# ──────────────────────────────────────────────────────────────────────────────


def process_documents(file_path: str):
    """
    Complete document processing pipeline
    """
    # 1. Load document
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # 3. Inspect chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk.page_content)}")
        print(f"Preview: {chunk.page_content[:150]}...")
        print(f"Metadata: {chunk.metadata}")

    return chunks


# Usage
chunks = process_documents(DATA_FILE)
