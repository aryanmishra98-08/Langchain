from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
PDF_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.pdf"  # path to the PDF file to load
PAGE_PREVIEW_LIMIT = 3                                     # how many pages to display in preview
CONTENT_PREVIEW_LENGTH = 150                               # characters to show in preview
# ──────────────────────────────────────────────────────────────────────────────

# Load a PDF file
loader = PyPDFLoader(str(PDF_FILE))
pages = loader.load()

print(f"Total pages: {len(pages)}")
for i, page in enumerate(pages[:PAGE_PREVIEW_LIMIT]):
    print(f"\n--- Page {i+1} ---")
    print(f"Content: {page.page_content[:CONTENT_PREVIEW_LENGTH]}...")
    print(f"Metadata: {page.metadata}")
