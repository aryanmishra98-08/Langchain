from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Load a single text file
base_dir = Path(__file__).resolve().parents[1]
pdf_file_path = base_dir / "data" / "TheFrenchRevolution.pdf"
loader = PyPDFLoader(str(pdf_file_path))
pages = loader.load()

print(f"Total pages: {len(pages)}")
for i, page in enumerate(pages[:3]):
    print(f"\n--- Page {i+1} ---")
    print(f"Content: {page.page_content[:150]}...")
    print(f"Metadata: {page.metadata}")
