from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
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
base_dir = Path(__file__).resolve().parents[1]
file_path = base_dir / "data" / "TheFrenchRevolution.txt"
chunks = process_documents(file_path)
