from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = _BASE_DIR / "data"  # directory to load files from
GLOB_PATTERN = "*.txt"         # file pattern to match (e.g. "*.txt", "*.pdf")
DOC_PREVIEW_LIMIT = 3          # how many documents to display in preview
CONTENT_PREVIEW_LENGTH = 150   # characters to show in preview
# ──────────────────────────────────────────────────────────────────────────────

# Load all matching files from directory
loader = DirectoryLoader(
    DATA_DIR,
    glob=GLOB_PATTERN,
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True
)
docs = loader.load()

print(f"Loaded {len(docs)} documents from directory")
for i, doc in enumerate(docs[:DOC_PREVIEW_LIMIT]):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content[:CONTENT_PREVIEW_LENGTH]}...")
    print(f"Metadata: {doc.metadata}")
