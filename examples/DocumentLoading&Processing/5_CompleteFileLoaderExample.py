# Complete Loading Example
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = _BASE_DIR / "data"  # directory containing documents to load
PDF_GLOB = "*.pdf"             # glob pattern for PDF files
TXT_GLOB = "*.txt"             # glob pattern for text files
FILE_ENCODING = "utf-8"        # encoding for text files
DOC_PREVIEW_LIMIT = 3          # how many documents to display in preview
CONTENT_PREVIEW_LENGTH = 150   # characters to show in preview
# ──────────────────────────────────────────────────────────────────────────────


def load_documents(source_dir) -> list:
    """
    Load documents from multiple sources
    """
    all_docs = []

    # Load PDFs
    pdf_loader = DirectoryLoader(
        source_dir,
        glob=PDF_GLOB,
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    pdf_docs = pdf_loader.load()
    all_docs.extend(pdf_docs)

    # Load text files
    txt_loader = DirectoryLoader(
        source_dir,
        glob=TXT_GLOB,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": FILE_ENCODING},
        show_progress=True
    )
    txt_docs = txt_loader.load()
    all_docs.extend(txt_docs)

    print(f"✓ Loaded {len(pdf_docs)} PDFs")
    print(f"✓ Loaded {len(txt_docs)} text files")
    print(f"✓ Total documents: {len(all_docs)}")
    for i, doc in enumerate(all_docs[:DOC_PREVIEW_LIMIT]):
        print(f"\n--- Document {i+1} ---")
        print(f"Content: {doc.page_content[:CONTENT_PREVIEW_LENGTH]}...")
        print(f"Metadata: {doc.metadata}")
    return all_docs


# Usage
documents = load_documents(DATA_DIR)
