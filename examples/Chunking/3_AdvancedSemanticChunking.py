from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Load a single text file
base_dir = Path(__file__).resolve().parents[1]
file_path = base_dir / "data" / "TheFrenchRevolution.txt"
loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()

# Split based on semantic similarity
semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95,
)

semantic_chunks = semantic_splitter.split_documents(documents)
print(f"Semantic chunks: {len(semantic_chunks)}")
print(
    f"Sample semantic chunk content: {semantic_chunks[0].page_content[:200]}")
