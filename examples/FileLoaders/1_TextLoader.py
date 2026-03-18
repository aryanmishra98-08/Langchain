from pathlib import Path
from langchain_community.document_loaders import TextLoader

# Load a single text file
base_dir = Path(__file__).resolve().parents[1]
file_path = base_dir / "data" / "TheFrenchRevolution.txt"
loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Content preview: {documents[0].page_content[:200]}")
print(f"Metadata: {documents[0].metadata}")
