from langchain_community.document_loaders import WebBaseLoader

# Load from URL
loader = WebBaseLoader([
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/concepts/",
])
docs = loader.load()

print(f"Loaded {len(docs)} web page(s)")
print(f"Content preview: {docs[0].page_content[:200]}...")
print(f"Metadata: {docs[0].metadata}")
