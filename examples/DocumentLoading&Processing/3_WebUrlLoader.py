from langchain_community.document_loaders import WebBaseLoader

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
URLS = [                                              # list of URLs to load
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/concepts/",
]
CONTENT_PREVIEW_LENGTH = 200                          # characters to show in preview
# ──────────────────────────────────────────────────────────────────────────────

# Load from URLs
loader = WebBaseLoader(URLS)
docs = loader.load()

print(f"Loaded {len(docs)} web page(s)")
print(f"Content preview: {docs[0].page_content[:CONTENT_PREVIEW_LENGTH]}...")
print(f"Metadata: {docs[0].metadata}")
