from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader

# Load all text files from directory
base_dir = Path(__file__).resolve().parents[1]
loader = DirectoryLoader(
    base_dir / "data",
    glob="*.txt",
    show_progress=True,
    use_multithreading=True
)
docs = loader.load()

print(f"Loaded {len(docs)} documents from directory")
for i, doc in enumerate(docs[:3]):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content[:150]}...")
    print(f"Metadata: {doc.metadata}")
