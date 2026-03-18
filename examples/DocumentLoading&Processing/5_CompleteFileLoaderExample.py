# Complete Loading Example
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

load_dotenv()


def load_documents(source_dir: str) -> list:
    """
    Load documents from multiple sources
    """
    all_docs = []

    # Load PDFs
    pdf_loader = DirectoryLoader(
        source_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    pdf_docs = pdf_loader.load()
    all_docs.extend(pdf_docs)

    # Load text files
    txt_loader = DirectoryLoader(
        source_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    txt_docs = txt_loader.load()
    all_docs.extend(txt_docs)

    print(f"✓ Loaded {len(pdf_docs)} PDFs")
    print(f"✓ Loaded {len(txt_docs)} text files")
    print(f"✓ Total documents: {len(all_docs)}")
    for i, doc in enumerate(all_docs[:3]):
        print(f"\n--- Document {i+1} ---")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")
    return all_docs


# Usage
documents = load_documents(Path(__file__).resolve().parents[1] / "data")
