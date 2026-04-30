# LangChain RAG: Comprehensive Intermediate Learning Guide

**Target Audience:** Developers with LangChain basics and LLM invocation experience  
**Duration:** 8-12 hours of hands-on learning  
**Prerequisites:** Python 3.8+, Basic LangChain knowledge, Understanding of LLMs

---

## Table of Contents

1. [RAG Fundamentals](#1-rag-fundamentals)
2. [Vector Embeddings Deep Dive](#2-vector-embeddings-deep-dive)
3. [Document Loading & Processing](#3-document-loading--processing)
4. [Text Chunking Strategies](#4-text-chunking-strategies)
5. [Vector Stores & Similarity Search](#5-vector-stores--similarity-search)
6. [Building RAG Pipelines](#6-building-rag-pipelines)
7. [Advanced RAG Patterns](#7-advanced-rag-patterns)
8. [Intermediate Project: PDF Q&A Chatbot](#8-intermediate-project-pdf-qa-chatbot)
9. [Performance Optimization](#9-performance-optimization)
10. [Exercises & Challenges](#10-exercises--challenges)
11. [Troubleshooting (macOS)](#11-troubleshooting-macos)

---

## Project Structure

```
Langchain/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── keys/
│   └── .env                           # Azure OpenAI credentials (loaded by every example)
├── myenv/                             # Python 3.11 virtual environment
├── examples/
│   ├── data/
│   │   ├── TheFrenchRevolution.txt    # Sample dataset (text)
│   │   ├── TheFrenchRevolution.pdf    # Sample dataset (PDF)
│   │   └── TheFrenchRevolution.docx   # Sample dataset (Word)
│   ├── DocumentLoading&Processing/    # Section 3 examples
│   │   ├── 1_TextLoader.py
│   │   ├── 2_PDFLoader.py
│   │   ├── 3_WebUrlLoader.py
│   │   ├── 4_DirectoryLoader.py
│   │   └── 5_CompleteFileLoaderExample.py
│   ├── TextChunkingStrategies/        # Section 4 examples
│   │   ├── 1_Chunking.py
│   │   ├── 2_OptimalChunkingParameters.py
│   │   ├── 3_AdvancedSemanticChunking.py
│   │   └── 4_PracticalChunkingExample.py
│   ├── VectorStores&SimilaritySearch/ # Section 5 examples
│   │   ├── 1_ChromaDBSetup&SampleSearch.py
│   │   └── 2_CompleteVectorStoreExample.py
│   ├── BuildingRAGPipelines/          # Section 6 examples
│   │   ├── 1_BasicRAGChain.py
│   │   ├── 2_ModernRAGwithLCEL.py
│   │   ├── 3_RAGwithSourceCitations.py
│   │   ├── 4_MultiQueryRAG.py
│   │   ├── 5_ConversationalRAG.py
│   │   └── 6_CompleteProductionReadyRAGSystem.py
│   └── CompleteRAGImplementation.py   # Section 8 full project
```

> **Sample data:** The repo ships with `TheFrenchRevolution.{txt,pdf,docx}` so every example runs out of the box once your `keys/.env` is configured.

---

## Setup

This project uses **Azure OpenAI** for both chat completion and embeddings. You'll need an Azure OpenAI resource with deployments for a chat model (e.g. `gpt-4o`) and an embeddings model (e.g. `text-embedding-3-small`).

```bash
# 1. Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create keys/.env with your Azure OpenAI credentials
mkdir -p keys
cat > keys/.env <<'EOF'
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=your-chat-deployment-name
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=your-embeddings-deployment-name
EOF
```

Every example loads this file via `load_dotenv(dotenv_path=… / "keys" / ".env")`, so the credentials only need to live in one place.

### Configuration blocks

Each example script has a clearly marked `# ── CONFIGURATION ──` block at the top with editable constants (data file path, chunk size, demo queries, etc.). To adapt an example to your own use case, edit those constants — you generally don't need to touch the rest of the file.

---

## 1. RAG Fundamentals

### 1.1 Why RAG?

**The Problem: LLM Hallucinations**

```
User: "What was our company's Q3 revenue?"
LLM (without RAG): "Based on typical industry patterns, I estimate..."
❌ WRONG: The LLM is making up information
```

**The Solution: Retrieval-Augmented Generation**

```
User: "What was our company's Q3 revenue?"
System:
  1. Retrieve: Search company documents → Find: "Q3 revenue: $45M"
  2. Augment: Add retrieved context to prompt
  3. Generate: LLM answers: "According to Q3 report, revenue was $45M"
✅ CORRECT: Grounded in actual documents
```

### 1.2 RAG Architecture

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│  STEP 1: EMBEDDING GENERATION        │
│  Query → Vector Representation       │
│  "What is RAG?" → [0.2, 0.8, ...]    │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  STEP 2: SIMILARITY SEARCH           │
│  Find k most similar documents       │
│  Vector DB: Compare embeddings       │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  STEP 3: CONTEXT RETRIEVAL           │
│  Retrieved Docs: [Doc1, Doc2, Doc3]  │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  STEP 4: PROMPT AUGMENTATION         │
│  Combine: Query + Retrieved Context  │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  STEP 5: LLM GENERATION              │
│  Generate answer using context       │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────┐
│    Answer    │
└──────────────┘
```

### 1.3 Key Components

1. **Document Loader**: Ingests various file formats
2. **Text Splitter**: Breaks documents into chunks
3. **Embedding Model**: Converts text to vectors
4. **Vector Store**: Stores and searches embeddings
5. **Retriever**: Fetches relevant documents
6. **LLM**: Generates final answer

---

## 2. Vector Embeddings Deep Dive

### 2.1 What Are Embeddings?

Embeddings are numerical representations of text that capture semantic meaning.

```python
# Conceptual example (simplified)
"dog" → [0.8, 0.2, 0.1, ...]  # 1536 dimensions
"puppy" → [0.7, 0.3, 0.15, ...] # Similar vector!
"car" → [0.1, 0.05, 0.9, ...]  # Different vector

# Similarity (cosine distance)
similarity("dog", "puppy") = 0.95  # High similarity
similarity("dog", "car") = 0.12    # Low similarity
```

### 2.2 Visual Representation

```
High-dimensional space (simplified to 2D):

                   cat •
                      /|\
                     / | \
                    /  |  \
           kitten •    |   • puppy
                  \    |    /
                   \   |   /
                    \  |  /
                     \ | /
                      \|/
                      dog •


                              • car
                             /
                            /
                           • truck
```

### 2.3 Embedding Models

**Popular Options:**
- **Azure OpenAI / OpenAI**: `text-embedding-3-small` (1536 dims), `text-embedding-3-large` (3072 dims). Legacy: `text-embedding-ada-002` (1536 dims).
- **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **Cohere**: `embed-english-v3.0`

> On Azure OpenAI you reference embedding models by **deployment name**, not model name. The examples in this repo read the deployment name from `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`.

**Cost vs Performance:**
- Larger dimensions = Better accuracy, Higher cost
- Smaller dimensions = Faster search, Lower accuracy

---

## 3. Document Loading & Processing

> **Examples:** `examples/DocumentLoading&Processing/`

### 3.1 Setup & Installation

All dependencies are pinned in [requirements.txt](requirements.txt) and installed in one step:

```bash
pip install -r requirements.txt
```

That installs:

```
# Core LangChain (v1.x)
langchain, langchain-classic, langchain-community, langchain-core
langchain-openai, langchain-chroma, langchain-text-splitters, langchain-experimental

# Vector store backend
chromadb

# Tokenization
tiktoken

# Document loaders
pypdf, beautifulsoup4

# Utilities
python-dotenv
```

### 3.2 Loading Different Document Types

#### 3.2.1 Text Files

> See: [examples/DocumentLoading&Processing/1_TextLoader.py](examples/DocumentLoading&Processing/1_TextLoader.py)

```python
from pathlib import Path
from langchain_community.document_loaders import TextLoader

DATA_FILE = Path("examples/data/TheFrenchRevolution.txt")

loader = TextLoader(str(DATA_FILE), encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Content preview: {documents[0].page_content[:200]}")
print(f"Metadata: {documents[0].metadata}")
```

#### 3.2.2 PDF Documents

> See: [examples/DocumentLoading&Processing/2_PDFLoader.py](examples/DocumentLoading&Processing/2_PDFLoader.py)

```python
from langchain_community.document_loaders import PyPDFLoader

# Load PDF with page-level granularity
loader = PyPDFLoader("examples/data/TheFrenchRevolution.pdf")
pages = loader.load()

print(f"Total pages: {len(pages)}")
for i, page in enumerate(pages[:3]):
    print(f"\n--- Page {i+1} ---")
    print(f"Content: {page.page_content[:150]}...")
    print(f"Metadata: {page.metadata}")
```

#### 3.2.3 Web Pages

> See: [examples/DocumentLoading&Processing/3_WebUrlLoader.py](examples/DocumentLoading&Processing/3_WebUrlLoader.py)

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader([
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/concepts/",
])
docs = loader.load()

print(f"Loaded {len(docs)} web page(s)")
```

#### 3.2.4 Directory Loader (Multiple Files)

> See: [examples/DocumentLoading&Processing/4_DirectoryLoader.py](examples/DocumentLoading&Processing/4_DirectoryLoader.py)

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "examples/data",
    glob="*.txt",
    show_progress=True,
    use_multithreading=True,
)
docs = loader.load()

print(f"Loaded {len(docs)} documents from directory")
```

### 3.3 Complete Loading Example

> See: [examples/DocumentLoading&Processing/5_CompleteFileLoaderExample.py](examples/DocumentLoading&Processing/5_CompleteFileLoaderExample.py)

```python
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")


def load_documents(source_dir) -> list:
    """Load PDFs and text files from a directory."""
    all_docs = []

    pdf_loader = DirectoryLoader(
        source_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    all_docs.extend(pdf_loader.load())

    txt_loader = DirectoryLoader(
        source_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    all_docs.extend(txt_loader.load())

    print(f"✓ Total documents: {len(all_docs)}")
    return all_docs


documents = load_documents(Path(__file__).resolve().parents[1] / "data")
```

---

## 4. Text Chunking Strategies

> **Examples:** `examples/TextChunkingStrategies/`

### 4.1 Why Chunking?

**Problems with whole documents:**
- Too large for embedding models (token limits)
- Diluted relevance (mixing multiple topics)
- Inefficient retrieval

**Benefits of chunking:**
- Precise retrieval
- Better semantic matching
- Manageable context size

### 4.2 Chunking Strategies Compared

> See: [examples/TextChunkingStrategies/1_Chunking.py](examples/TextChunkingStrategies/1_Chunking.py)

```python
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# Sample text
sample_text = """
Artificial Intelligence (AI) has revolutionized many industries.
Machine learning is a subset of AI.

Deep learning uses neural networks with multiple layers.
It has achieved breakthrough results in computer vision.

Natural Language Processing enables computers to understand human language.
"""

# Strategy 1: Character-based splitting
char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)
char_chunks = char_splitter.split_text(sample_text)

# Strategy 2: Recursive splitting (RECOMMENDED)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(sample_text)

# Strategy 3: Token-based splitting
token_splitter = TokenTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)
token_chunks = token_splitter.split_text(sample_text)

print("Character-based chunks:", len(char_chunks))
print("Recursive chunks:", len(recursive_chunks))
print("Token-based chunks:", len(token_chunks))
```

### 4.3 Optimal Chunking Parameters

> See: [examples/TextChunkingStrategies/2_OptimalChunkingParameters.py](examples/TextChunkingStrategies/2_OptimalChunkingParameters.py)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_optimized_splitter(use_case: str):
    """
    Return optimized splitter based on use case
    """
    configs = {
        "general": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "code": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "separators": ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "],
        },
        "qa": {
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        "long_context": {
            "chunk_size": 2000,
            "chunk_overlap": 400,
        }
    }
    
    config = configs.get(use_case, configs["general"])
    
    return RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
        separators=config.get("separators", ["\n\n", "\n", " ", ""]),
    )

# Usage
splitter = create_optimized_splitter("qa")
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
```

### 4.4 Advanced: Semantic Chunking

> See: [examples/TextChunkingStrategies/3_AdvancedSemanticChunking.py](examples/TextChunkingStrategies/3_AdvancedSemanticChunking.py)

```python
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings

# Split based on semantic similarity
semantic_splitter = SemanticChunker(
    AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    ),
    breakpoint_threshold_type="percentile",  # also: "standard_deviation", "interquartile", "gradient"
    breakpoint_threshold_amount=95,
)

semantic_chunks = semantic_splitter.split_documents(documents)
print(f"Semantic chunks: {len(semantic_chunks)}")
```

### 4.5 Practical Chunking Example

> See: [examples/TextChunkingStrategies/4_PracticalChunkingExample.py](examples/TextChunkingStrategies/4_PracticalChunkingExample.py)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


def process_documents(file_path: str):
    """Load → split → inspect."""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk.page_content)}")
        print(f"Preview: {chunk.page_content[:150]}...")
        print(f"Metadata: {chunk.metadata}")

    return chunks


chunks = process_documents("examples/data/TheFrenchRevolution.txt")
```

---

## 5. Vector Stores & Similarity Search

> **Examples:** `examples/VectorStores&SimilaritySearch/`

### 5.1 Vector Store Options

| Vector Store | Use Case | Pros | Cons |
|-------------|----------|------|------|
| **Chroma** | Development, Small datasets | Easy setup, Free | Not production-scale |
| **Pinecone** | Production, Large scale | Managed, Fast | Paid service |
| **Weaviate** | Production, Self-hosted | Feature-rich, Free tier | Complex setup |
| **FAISS** | Local, Fast prototyping | Very fast, Free | In-memory only |
| **Qdrant** | Production, Self-hosted | Modern, Efficient | Newer option |

### 5.2 Chroma DB Setup (Recommended for Learning)

> See: [examples/VectorStores&SimilaritySearch/1_ChromaDBSetup&SampleSearch.py](examples/VectorStores&SimilaritySearch/1_ChromaDBSetup&SampleSearch.py)

```python
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="the_french_revolution",
)

doc_count = len(vectorstore.get()["ids"])
print(f"✓ Created vector store with {doc_count} documents")
```

### 5.3 Similarity Search

```python
query = "What caused the French Revolution?"
results = vectorstore.similarity_search(query, k=3)

print(f"Found {len(results)} relevant documents:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}\n")
```

### 5.4 Similarity Search with Scores

```python
query = "What happened during the Reign of Terror?"
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:150]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
```

### 5.5 Maximum Marginal Relevance (MMR)

```python
# MMR: Balances relevance with diversity
results_mmr = vectorstore.max_marginal_relevance_search(
    query,
    k=4,
    fetch_k=20,
    lambda_mult=0.5  # 0=max diversity, 1=max relevance
)

print(f"MMR returned {len(results_mmr)} diverse results")
```

### 5.6 Complete Vector Store Example

> See: [examples/VectorStores&SimilaritySearch/2_CompleteVectorStoreExample.py](examples/VectorStores&SimilaritySearch/2_CompleteVectorStoreExample.py)

```python
import os
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class DocumentVectorStore:
    """Manages document vectorization and retrieval."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        )
        self.vectorstore: Chroma | None = None

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"✓ Vector store created with {len(chunks)} chunks")
        return self.vectorstore

    def load_existing(self) -> Chroma:
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        return self.vectorstore

    def search(self, query: str, k: int = 3, method: str = "similarity"):
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        if method == "similarity":
            return self.vectorstore.similarity_search(query, k=k)
        if method == "mmr":
            return self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=k * 5
            )
        if method == "similarity_score":
            return self.vectorstore.similarity_search_with_score(query, k=k)
        raise ValueError("Use 'similarity', 'mmr', or 'similarity_score'.")


if __name__ == "__main__":
    loader = TextLoader("examples/data/TheFrenchRevolution.txt", encoding="utf-8")
    docs = loader.load()

    vs = DocumentVectorStore()
    vs.create_from_documents(docs)

    results = vs.search("What were the main causes of the French Revolution?", k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:200])
```

---

## 6. Building RAG Pipelines

> **Examples:** `examples/BuildingRAGPipelines/`

### 6.1 Basic RAG Chain

> See: [examples/BuildingRAGPipelines/1_BasicRAGChain.py](examples/BuildingRAGPipelines/1_BasicRAGChain.py)

The legacy `RetrievalQA` class is deprecated. The v1 chain-based replacement (in `langchain-classic`) uses composable building blocks: `create_stuff_documents_chain` for the answer step and `create_retrieval_chain` to wire retrieval into it.

```python
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep the answer concise.\n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

result = rag_chain.invoke({"input": "What is the main topic of the document?"})
print("Answer:", result["answer"])
for doc in result["context"]:
    print(f"- {doc.metadata.get('source', 'Unknown')}")
```

### 6.2 Modern RAG with LCEL

> See: [examples/BuildingRAGPipelines/2_ModernRAGwithLCEL.py](examples/BuildingRAGPipelines/2_ModernRAGwithLCEL.py)

For maximum control, build the chain by hand with LCEL primitives.

```python
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: Provide a detailed answer based on the context. If the answer cannot be found
in the context, say "I cannot find this information in the provided documents."
"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What were the main causes of the French Revolution?")
print(response)
```

### 6.3 RAG with Source Citations

> See: [examples/BuildingRAGPipelines/3_RAGwithSourceCitations.py](examples/BuildingRAGPipelines/3_RAGwithSourceCitations.py)

```python
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """Answer the question based on the following context.
After your answer, list the sources you used with their page numbers.

Context:
{context}

Question: {question}

Answer format:
[Your detailed answer]

Sources:
- [Source 1 with page number]
- [Source 2 with page number]
"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Document {i+1}] (Source: {source}, Page: {page})\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


rag_chain_with_sources = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain_with_sources.invoke("Explain the key concepts")
print(response)
```

### 6.4 Multi-Query RAG (Advanced)

> See: [examples/BuildingRAGPipelines/4_MultiQueryRAG.py](examples/BuildingRAGPipelines/4_MultiQueryRAG.py)

```python
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Multi-query retriever generates several search queries from one user query
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

unique_docs = multi_query_retriever.invoke(
    "What were the main causes of the French Revolution?"
)
print(f"Retrieved {len(unique_docs)} unique documents")
```

### 6.5 Conversational RAG (With Memory)

> See: [examples/BuildingRAGPipelines/5_ConversationalRAG.py](examples/BuildingRAGPipelines/5_ConversationalRAG.py)

The legacy `ConversationalRetrievalChain` + `ConversationBufferMemory` API is deprecated. The v1 chain-based replacement (in `langchain-classic`) is **`create_history_aware_retriever`**, which rewrites follow-up questions into standalone queries before retrieval. Chat history is kept as a plain list of `HumanMessage`/`AIMessage`.

```python
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Step 1 — rewrite the latest question into a standalone query using chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference "
     "context in the chat history, formulate a standalone question which can be "
     "understood without the chat history. Do NOT answer the question, just "
     "reformulate it if needed; otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Step 2 — answer using retrieved context + chat history
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for question-answering tasks. "
     "Use the following pieces of retrieved context to answer the question. "
     "If you don't know the answer, just say that you don't know. "
     "Keep the answer concise.\n\n"
     "Context: {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Multi-turn conversation — chat history is just a list
chat_history = []
queries = [
    "What were the main causes of the French Revolution?",
    "What were the major events during the Revolution?",   # uses prior context
    "Can you explain more about the first event?",         # references prior answer
]

for query in queries:
    print(f"\nUser: {query}")
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    answer = result["answer"]
    print(f"Assistant: {answer}")
    chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer),
    ])
```

### 6.6 Complete Production-Ready RAG System

> See: [examples/BuildingRAGPipelines/6_CompleteProductionReadyRAGSystem.py](examples/BuildingRAGPipelines/6_CompleteProductionReadyRAGSystem.py)

This pattern uses `RunnableParallel` + `.assign()` so retrieval runs once per query and both the answer and source documents come back in a single pass.

```python
import os
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma


class ProductionRAGSystem:
    """Production-ready RAG system with best practices."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        temperature: float = 0.0,
        k: int = 4,
        streaming: bool = True,
    ):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=temperature,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )
        self.chain = self._build_chain()

    def _build_chain(self):
        template = """You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Provide a comprehensive answer based on the context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific parts of the context when possible
4. Be concise but thorough

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(
                f"[Source {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )

        answer_chain = prompt | self.llm | StrOutputParser()

        # Retrieve once, then run the answer chain alongside the source docs.
        return RunnableParallel(
            {
                "source_documents": self.retriever,
                "question": RunnablePassthrough(),
            }
        ).assign(
            answer=lambda x: answer_chain.invoke(
                {"context": format_docs(x["source_documents"]), "question": x["question"]}
            )
        )

    def query(self, question: str) -> Dict[str, Any]:
        result = self.chain.invoke(question)
        return {
            "answer": result["answer"],
            "source_documents": [
                {"content": doc.page_content[:200] + "...", "metadata": doc.metadata}
                for doc in result["source_documents"]
            ],
            "num_sources": len(result["source_documents"]),
        }

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        return [self.query(q) for q in questions]


if __name__ == "__main__":
    rag = ProductionRAGSystem(persist_directory="./chroma_db", k=3)
    result = rag.query("What is the main topic discussed?")
    print("\n\nAnswer:", result["answer"])
    print(f"\nUsed {result['num_sources']} sources")
```

---

## 7. Advanced RAG Patterns

### 7.1 Retriever Comparison

```python
import os
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 1. Similarity search
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# 2. MMR (Maximum Marginal Relevance) — diverse results
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10},
)

# 3. Similarity with threshold
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 3},
)

query = "causes of the French Revolution"

print("Similarity Search:")
print(f"Found {len(similarity_retriever.invoke(query))} documents\n")

print("MMR Search (Diverse):")
print(f"Found {len(mmr_retriever.invoke(query))} documents\n")

print("Threshold Search:")
print(f"Found {len(threshold_retriever.invoke(query))} documents")
```

### 7.2 Contextual Compression

```python
import os
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import AzureChatOpenAI

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

compressed_docs = compression_retriever.invoke(
    "What were the most important events of the Reign of Terror?"
)
print(f"Retrieved {len(compressed_docs)} compressed documents")
for doc in compressed_docs:
    print(f"\n{doc.page_content}")
```

### 7.3 Parent Document Retriever

```python
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

store = InMemoryStore()

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

parent_retriever.add_documents(documents)

# Retrieves small chunks for matching but returns the larger parent for context
docs = parent_retriever.invoke("Who were the key figures of the Revolution?")
```

### 7.4 Self-Query Retriever

```python
import os
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_openai import AzureChatOpenAI

metadata_field_info = [
    AttributeInfo(name="source", description="The source document name", type="string"),
    AttributeInfo(name="page", description="The page number in the source document", type="integer"),
]

document_content_description = "Historical documentation about the French Revolution"

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0,
)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
)

# Natural language query that includes a metadata filter
docs = retriever.invoke("What does the document say about the Reign of Terror on page 5?")
```

### 7.5 Ensemble Retriever

```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Vector retriever (semantic)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# BM25 retriever (keyword)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

docs = ensemble_retriever.invoke("Bastille storming")
print(f"Ensemble retrieved {len(docs)} documents")
```

---

## 8. Intermediate Project: PDF Q&A Chatbot

> **Full implementation:** [examples/CompleteRAGImplementation.py](examples/CompleteRAGImplementation.py)

### 8.1 Project Overview

**Goal:** Build a chatbot that answers questions about a directory of PDFs.

**Features:**
- Loads every PDF in a directory
- Persistent ChromaDB vector store (reused across runs)
- Conversational memory via history-aware retriever
- MMR-based retrieval for diverse context
- Source citations on every answer
- Interactive REPL with `quit` / `reset` commands

### 8.2 Implementation Sketch

The full ~310-line implementation lives in [examples/CompleteRAGImplementation.py](examples/CompleteRAGImplementation.py). Below is the core wiring — it's the same pattern as §6.5, applied to a directory of PDFs.

```python
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "keys" / ".env")


class PDFChatbot:
    """Chatbot for answering questions about a directory of PDF documents."""

    def __init__(
        self,
        pdf_directory: str,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4,
    ):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k

        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=0,
        )
        self.vectorstore: Chroma | None = None
        self.chain = None
        self.chat_history: List[BaseMessage] = []

    def initialize(self) -> None:
        # 1. Load PDFs
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        )
        documents = loader.load()

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        # 3. Create or load the vector store
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            if not self.vectorstore.get()["ids"]:
                self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

        # 4. Build the conversational RAG chain (history-aware retriever + QA)
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.k * 3},
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question which might reference "
             "context in the chat history, formulate a standalone question which can be "
             "understood without the chat history. Do NOT answer the question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Use the retrieved context to answer. "
             "If you don't know, say \"I cannot find this information in the provided documents.\"\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def ask(self, question: str) -> Dict[str, Any]:
        result = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
        })
        answer = result["answer"]
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer),
        ])
        return {
            "answer": answer,
            "sources": [
                {
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "Unknown"),
                }
                for doc in result["context"]
            ],
        }
```

> **Note:** the `result` returned by `create_retrieval_chain` puts the retrieved docs under `result["context"]` (not `result["source_documents"]` like the legacy chain).

### 8.3 Usage Instructions

```bash
# 1. Install dependencies (already covered by the top-level Setup)
pip install -r requirements.txt

# 2. Make sure keys/.env exists with your Azure OpenAI credentials
#    (see the top-level Setup section)

# 3. The repo ships with examples/data/TheFrenchRevolution.pdf;
#    drop additional PDFs into examples/data/ if you want.

# 4. Run the chatbot
python examples/CompleteRAGImplementation.py
```

### 8.4 Example Conversation

```
============================================================
PDF Q&A CHATBOT
============================================================
Ask questions about your PDF documents!
Type 'quit', 'exit', or 'q' to end the conversation
Type 'reset' to clear conversation history
============================================================

You: What were the main causes of the French Revolution?

🤖 Assistant: The main causes were widespread financial crisis driven by
costly wars and royal extravagance, deep social inequality between the
estates, food shortages caused by poor harvests, and Enlightenment ideas
that challenged the legitimacy of absolute monarchy.

📚 Sources:
   1. TheFrenchRevolution.pdf (Page 1)
   2. TheFrenchRevolution.pdf (Page 2)

You: What about the Reign of Terror specifically?

🤖 Assistant: The Reign of Terror (1793–1794) was a period in which the
revolutionary government, dominated by the Committee of Public Safety
under Robespierre, used mass executions to suppress perceived enemies of
the Revolution...

📚 Sources:
   1. TheFrenchRevolution.pdf (Page 4)

You: quit

👋 Goodbye!
```

---

## 9. Performance Optimization

### 9.1 Chunk Size Optimization

```python
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

def benchmark_chunk_sizes(documents, sizes=[500, 1000, 1500, 2000]):
    """Test different chunk sizes"""
    results = {}
    
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 5  # 20% overlap
        )
        
        start = time.time()
        chunks = splitter.split_documents(documents)
        elapsed = time.time() - start
        
        results[size] = {
            "num_chunks": len(chunks),
            "time": elapsed,
            "avg_chunk_size": sum(len(c.page_content) for c in chunks) / len(chunks)
        }
    
    print("Chunk Size Benchmark:")
    print(f"{'Size':<10} {'Chunks':<10} {'Time (s)':<12} {'Avg Size':<10}")
    print("-" * 50)
    for size, data in results.items():
        print(f"{size:<10} {data['num_chunks']:<10} {data['time']:<12.3f} {data['avg_chunk_size']:<10.0f}")
    
    return results

results = benchmark_chunk_sizes(documents)
```

### 9.2 Embedding Model Comparison

> Note: HuggingFace embeddings require `pip install sentence-transformers` (not in `requirements.txt` by default).

```python
import os
import time
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def compare_embedding_models(texts):
    """Compare different embedding models on the same input."""
    models = {
        "Azure OpenAI": AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        ),
        "HuggingFace (MiniLM)": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
        "HuggingFace (MPNet)": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
        ),
    }

    results = {}
    for name, model in models.items():
        start = time.time()
        embeddings = model.embed_documents(texts[:100])
        elapsed = time.time() - start
        results[name] = {
            "time": elapsed,
            "dimension": len(embeddings[0]),
            "cost": "Paid" if "Azure" in name else "Free (local)",
        }

    print("\nEmbedding Model Comparison:")
    print(f"{'Model':<30} {'Time (s)':<12} {'Dimensions':<12} {'Cost':<14}")
    print("-" * 70)
    for name, data in results.items():
        print(f"{name:<30} {data['time']:<12.3f} {data['dimension']:<12} {data['cost']:<14}")

    return results


sample_texts = [chunk.page_content for chunk in chunks[:100]]
compare_embedding_models(sample_texts)
```

### 9.3 Retrieval Optimization

```python
def optimize_retrieval_params(vectorstore, test_queries, k_values=(3, 5, 7, 10)):
    """Test different k values for retrieval."""
    print("\nRetrieval Parameter Optimization:")
    print(f"{'k Value':<10} {'Avg Docs':<12} {'Avg Relevance':<15}")
    print("-" * 40)

    for k in k_values:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.5},
        )

        total_docs = 0
        total_relevance = 0

        for query in test_queries:
            docs = retriever.invoke(query)
            total_docs += len(docs)
            total_relevance += len(docs) * 0.8

        avg_docs = total_docs / len(test_queries)
        avg_relevance = total_relevance / len(test_queries)

        print(f"{k:<10} {avg_docs:<12.1f} {avg_relevance:<15.2f}")


test_queries = [
    "What were the main causes of the French Revolution?",
    "What happened during the Reign of Terror?",
    "Who were the major figures of the Revolution?",
]
optimize_retrieval_params(vectorstore, test_queries)
```

### 9.4 Caching Strategies

```python
import os
import time
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_openai import AzureChatOpenAI

# Enable an LLM-level cache shared across all chat models
set_llm_cache(InMemoryCache())

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
)

# First call — hits the model
start = time.time()
llm.invoke("What is 2+2?")
time1 = time.time() - start

# Second call — served from the cache
start = time.time()
llm.invoke("What is 2+2?")
time2 = time.time() - start

print(f"First call: {time1:.3f}s")
print(f"Cached call: {time2:.3f}s")
print(f"Speedup: {time1/time2:.1f}x")
```

### 9.5 Batch Processing

```python
def batch_process_queries(rag_chain, queries, batch_size=5):
    """Process queries in batches for efficiency"""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        batch_results = rag_chain.batch(batch)
        results.extend(batch_results)
    
    return results

queries = [
    "What is ML?",
    "Explain DL",
    "What is NLP?",
    "Define AI",
    "What is computer vision?"
]
results = batch_process_queries(rag_chain, queries, batch_size=2)
```

---

## 10. Exercises & Challenges

### Exercise 1: Alternative Text Splitters

**Task:** Implement and compare three different text splitters.

```python
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# Your code here:
# 1. Load a document (e.g. examples/data/TheFrenchRevolution.txt)
# 2. Split using three different splitters
# 3. Compare the number and quality of chunks
# 4. Determine which is best for your use case
```

> **Optional:** for a fourth splitter, install spaCy (`pip install spacy && python -m spacy download en_core_web_sm`) and try `SpacyTextSplitter` from `langchain_text_splitters`.

**Expected Output:**
- Comparison table showing chunks created by each splitter
- Analysis of which splitter preserves semantic meaning best

---

### Exercise 2: Custom Metadata

**Task:** Add custom metadata to documents for better filtering

```python
from langchain_community.document_loaders import PyPDFLoader

def load_with_custom_metadata(file_path, category, author):
    """Load document and add custom metadata"""
    # Your implementation here
    pass

docs = load_with_custom_metadata("paper.pdf", category="research", author="John Doe")
```

---

### Exercise 3: Hybrid Search

**Task:** Implement a hybrid retriever combining semantic and keyword search

```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def create_hybrid_retriever(documents, vector_weight=0.5):
    """
    Create hybrid retriever
    
    Args:
        documents: List of documents
        vector_weight: Weight for vector search (0-1)
    Returns:
        EnsembleRetriever
    """
    # Your implementation here
    pass
```

---

### Exercise 4: Multi-Document Types

**Task:** Build a RAG system that handles PDFs, Word docs, and web pages

```python
def load_multi_format_documents(directory):
    """Load documents of different formats (PDF, DOCX, TXT, MD)"""
    # Your implementation here
    pass
```

---

### Exercise 5: Source Citation Enhancement

**Task:** Enhance the chatbot to provide inline citations

```python
def format_answer_with_citations(answer, sources):
    """
    Format answer with [1], [2] style citations
    
    Example output:
    "Machine learning is a subset of AI [1]. It involves 
    training models on data [2]."
    """
    # Your implementation here
    pass
```

---

### Exercise 6: Evaluation Metrics

**Task:** Implement evaluation metrics for your RAG system

```python
def evaluate_rag_system(rag_chain, test_set):
    """
    Evaluate RAG performance
    
    Metrics: Answer relevance, Faithfulness, Context precision, Context recall
    """
    # Your implementation here
    pass

test_set = [
    ("What were the main causes of the French Revolution?", "Financial crisis, social inequality..."),
    ("Who was Robespierre?", "A leader of the Jacobins during the Reign of Terror..."),
]
metrics = evaluate_rag_system(rag_chain, test_set)
```

---

### Challenge 1: Multi-language Support

Extend the chatbot to detect query language, use appropriate embeddings, and generate answers in the query language.

### Challenge 2: Streaming Responses

Implement streaming responses using `StreamingStdOutCallbackHandler` for better UX.

### Challenge 3: Query Expansion

Generate expanded/alternative queries for better retrieval coverage.

### Challenge 4: Document Upload API

Create a FastAPI endpoint for uploading documents and querying the RAG system.

### Challenge 5: Advanced Filtering

Implement document filtering by date range, categories, and minimum relevance score.

---

## 11. Troubleshooting (macOS)

### Azure OpenAI Errors

**`openai.NotFoundError: The API deployment for this resource does not exist`**
- Cause: `AZURE_OPENAI_CHAT_DEPLOYMENT` or `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` doesn't match a deployment in your Azure resource. The variable holds the **deployment name** you set in Azure, not the underlying model name.
- Fix: Open the Azure AI Studio / portal, copy the deployment names for your chat + embeddings models, and update `keys/.env`.

**`openai.AuthenticationError: Incorrect API key`**
- Cause: `AZURE_OPENAI_API_KEY` is wrong or copy-pasted with whitespace.
- Fix: Re-copy the key from Azure portal → Keys and Endpoint.

**`openai.BadRequestError` mentioning an unsupported API version**
- Cause: `OPENAI_API_VERSION` doesn't support the feature/model you're using.
- Fix: Use a recent version such as `2024-10-21`.

**`KeyError` for `AZURE_OPENAI_*` variables**
- Cause: `keys/.env` not found, or `load_dotenv()` ran before the file existed. Every example resolves the path as `… / "keys" / ".env"` — run scripts from the project root or check the resolved path matches.

### SSL Certificate Errors

If you encounter `[SSL: CERTIFICATE_VERIFY_FAILED]` errors when loading web URLs or using APIs:

**1. Confirm your Python build:**

```bash
source myenv/bin/activate
python -c "import sys,ssl; print(sys.executable); print(ssl.OPENSSL_VERSION)"
```

**2. Run the certificate installer (Python.org installs):**

```bash
open "/Applications/Python 3.11/Install Certificates.command"
```

**3. Upgrade certifi inside your venv:**

```bash
pip install --upgrade pip certifi
```

**4. Force Python/urllib to use the certifi bundle:**

```bash
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
```

**5. Verify HTTPS trust works:**

```bash
python -c "import urllib.request; print(urllib.request.urlopen('https://github.com').status)"
```

To persist this across sessions, add the two `export` lines to your shell profile (`~/.zshrc`) or to the venv's `activate` script.

### libmagic Warning

If you see `libmagic is unavailable but assists in filetype detection.`:
- This is a **warning only**, not an error — your script still works.
- The `Unstructured` library uses `libmagic` for file type detection; without it, detection may be less accurate.

**To suppress the warning:**

```bash
brew install libmagic
pip install python-magic
```

---

## Additional Resources

### Official Documentation
- [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/retrieval)
- [LangChain Concepts](https://docs.langchain.com/oss/python/langchain/overview)
- [Vector Stores](https://docs.langchain.com/oss/python/integrations/vectorstores)
- [Retrievers](https://docs.langchain.com/oss/python/integrations/retrievers)

### Best Practices
1. **Chunking:** Start with 1000 chars, 200 overlap
2. **Retrieval:** Use k=3-5 for most queries
3. **Embeddings:** Azure OpenAI / OpenAI for quality, HuggingFace `sentence-transformers` for cost-free local
4. **Vector Store:** Chroma for dev, Pinecone / Weaviate / Qdrant for production
5. **Prompt Engineering:** Always instruct the model to cite sources and refuse when context is insufficient
6. **Modern APIs:** Prefer `create_retrieval_chain` + `create_history_aware_retriever` over the deprecated `RetrievalQA` / `ConversationalRetrievalChain`

### Common Pitfalls
- ❌ Chunks too large → Poor retrieval
- ❌ No overlap → Lost context between chunks
- ❌ k too low → Missing relevant info
- ❌ k too high → Noise in context
- ❌ No source citation → Hard to verify answers

### Performance Tips
- Use caching for repeated queries
- Batch process when possible
- Choose appropriate embedding dimensions
- Monitor token usage and costs
- Implement error handling and retries

---

## Next Steps

After completing this guide, explore:

1. **Advanced RAG Patterns** — RAPTOR, Graph RAG, HyDE (Hypothetical Document Embeddings)
2. **Production Considerations** — Monitoring, logging, cost optimization, scaling, security
3. **Specialized RAG** — Code documentation, legal docs, medical literature, customer support
4. **Evaluation & Testing** — RAGAS framework, human evaluation, A/B testing

---

## Conclusion

You've now learned:
- ✅ RAG fundamentals and architecture
- ✅ Document loading and processing
- ✅ Text chunking strategies
- ✅ Vector embeddings and similarity search
- ✅ Building production-ready RAG pipelines
- ✅ Advanced retrieval patterns
- ✅ Performance optimization

**Keep practicing and experimenting!** RAG is a rapidly evolving field, and hands-on experience is the best way to master it.

---

<p align="center">
  Built for learning. Designed for production readiness.<br>
  Licensed under Apache 2.0.
</p>