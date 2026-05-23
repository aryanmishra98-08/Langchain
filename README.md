# LangChain RAG: A Progressive Hands-On Guide

A structured, example-driven repository for learning Retrieval-Augmented Generation (RAG) with [LangChain](https://www.langchain.com/) — the leading open-source framework for building LLM-powered applications. Each example builds on the last, taking you from loading your first document to a production-ready conversational PDF chatbot.

**Target Audience:** Developers with basic LangChain and LLM experience  
**Duration:** 2–3 hours of hands-on learning  
**Prerequisites:** Python 3.11+, Azure OpenAI credentials, Familiarity with LLM concepts

---

## Philosophy

RAG is powerful, but assembling its components — loaders, splitters, embeddings, vector stores, and retrieval chains — can feel overwhelming without a clear progression. This repository distills the full RAG stack into **18 focused, runnable examples** organized into four progressive learning tracks. The guiding principles are:

- **Learn by doing.** Every concept is a standalone Python script you can run, modify, and experiment with immediately.
- **Progressive complexity.** Examples are ordered so that each one introduces exactly one new idea on top of what came before.
- **Production awareness.** The journey doesn't stop at a basic Q&A chain. The final examples cover multi-query retrieval, conversational memory, source citations, and a fully deployable chatbot class.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Structure](#repository-structure)
3. [What is RAG?](#what-is-rag)
4. [Track 1 — Document Loading and Processing](#track-1--document-loading-and-processing-examples-15)
5. [Track 2 — Text Chunking Strategies](#track-2--text-chunking-strategies-examples-14)
6. [Track 3 — Vector Stores and Similarity Search](#track-3--vector-stores-and-similarity-search-examples-12)
7. [Track 4 — Building RAG Pipelines](#track-4--building-rag-pipelines-examples-16)
8. [Core Concepts at a Glance](#core-concepts-at-a-glance)
9. [Navigating the Examples](#navigating-the-examples)
10. [Retrieval Strategy Comparison](#retrieval-strategy-comparison)
11. [Migrating from Legacy APIs](#migrating-from-legacy-apis)
12. [Additional Resources](#additional-resources)

---

## Quick Start

### Prerequisites

- Python 3.11 or later
- An Azure OpenAI resource with a chat deployment (e.g., `gpt-4o`) and an embeddings deployment (e.g., `text-embedding-ada-002`) — the examples are wired for Azure OpenAI by default, but swapping providers requires only changing the model class and credentials.

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd Langchain
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API keys

Copy `keys/.env.example` to `keys/.env` and fill in your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-10-21
```

All examples that require credentials load this file automatically via:

```python
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")
```

### 3. Run your first example

```bash
python "examples/DocumentLoading&Processing/1_TextLoader.py"
```

If you see `Loaded 1 document(s)` followed by a content preview, you're ready to go.

---

## Repository Structure

```
Langchain/
├── README.md                                       ← You are here
├── requirements.txt                                ← Python dependencies
├── LICENSE                                         ← Apache 2.0
├── keys/
│   ├── .env.example                                ← Credential template
│   └── .env                                        ← Your API keys (not committed)
└── examples/
    ├── data/
    │   ├── TheFrenchRevolution.txt                 ← Sample text document
    │   ├── TheFrenchRevolution.pdf                 ← Sample PDF document
    │   └── TheFrenchRevolution.docx                ← Sample Word document
    ├── DocumentLoading&Processing/
    │   ├── 1_TextLoader.py                         ← Load a plain text file
    │   ├── 2_PDFLoader.py                          ← Load a PDF page-by-page
    │   ├── 3_WebUrlLoader.py                       ← Load from web URLs
    │   ├── 4_DirectoryLoader.py                    ← Bulk-load a directory of files
    │   └── 5_CompleteFileLoaderExample.py          ← Multi-format loader pipeline
    ├── TextChunkingStrategies/
    │   ├── 1_Chunking.py                           ← Three core splitting strategies
    │   ├── 2_OptimalChunkingParameters.py          ← Use-case-tuned chunk configs
    │   ├── 3_AdvancedSemanticChunking.py           ← Embedding-based semantic splits
    │   └── 4_PracticalChunkingExample.py           ← End-to-end chunk pipeline
    ├── VectorStores&SimilaritySearch/
    │   ├── 1_ChromaDBSetup&SampleSearch.py         ← Create a Chroma store and search
    │   └── 2_CompleteVectorStoreExample.py         ← Reusable DocumentVectorStore class
    ├── BuildingRAGPipelines/
    │   ├── 1_BasicRAGChain.py                      ← Classic retrieval chain
    │   ├── 2_ModernRAGwithLCEL.py                  ← LCEL pipe-style RAG chain
    │   ├── 3_RAGwithSourceCitations.py             ← Answers with page-level citations
    │   ├── 4_MultiQueryRAG.py                      ← Multi-query retrieval
    │   ├── 5_ConversationalRAG.py                  ← History-aware follow-up questions
    │   └── 6_CompleteProductionReadyRAGSystem.py   ← ProductionRAGSystem class
    └── CompleteRAGImplementation.py                ← Capstone: PDF Q&A chatbot
```

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a pattern for building LLM applications that answer questions grounded in a specific set of documents rather than the model's training data alone. It works by retrieving the most relevant passages from a document store at query time and injecting them into the prompt as context.

### Why use RAG?

Without it, you would need to:
- Fine-tune a model every time your knowledge base changes
- Trust the model to recall facts it may have hallucinated or never seen
- Manually manage context windows and source attribution
- Build custom search infrastructure from scratch for every project

LangChain abstracts all of this into a unified, composable interface.

### Core building blocks

| Concept | Description |
|---------|-------------|
| **Document Loaders** | Ingest text from files (PDF, TXT, DOCX), URLs, and directories into `Document` objects |
| **Text Splitters** | Divide large documents into retrieval-friendly chunks with configurable size and overlap |
| **Embeddings** | Convert text chunks into dense vector representations for semantic search |
| **Vector Stores** | Index and persist embeddings; serve nearest-neighbor queries at runtime |
| **Retrievers** | Query a vector store and return the most relevant chunks for a given question |
| **RAG Chains** | Combine retriever + prompt + LLM into a single pipeline that answers grounded questions |

---

## Track 1 — Document Loading and Processing (Examples 1–5)

### Theory

At its core, every RAG pipeline starts by turning raw files into LangChain `Document` objects. A `Document` carries two fields: `page_content` (the raw text) and `metadata` (a dict of source information like file path, page number, or URL).

LangChain provides loader classes for every common source type. Each loader exposes the same two-method interface:

| Method | Returns | Use When |
|--------|---------|----------|
| `.load()` | `List[Document]` | You want all pages at once |
| `.lazy_load()` | `Iterator[Document]` | You need memory-efficient streaming for large corpora |

The `metadata` dict is automatically populated by each loader — for PDFs it includes `{"source": "path/to/file.pdf", "page": 0}`, for web pages it includes the URL and title. This metadata flows through the entire pipeline unchanged and is what powers source citations in Track 4.

Different loaders produce different document granularity:

| Loader | Granularity | Notes |
|--------|-------------|-------|
| `TextLoader` | One `Document` per file | Entire file as `page_content` |
| `PyPDFLoader` | One `Document` per page | `metadata["page"]` is zero-indexed |
| `WebBaseLoader` | One `Document` per URL | Strips HTML automatically |
| `DirectoryLoader` | One `Document` per matched file | Delegates to a per-file loader class |

---

### Example 1 — Text Loader

**File:** `examples/DocumentLoading&Processing/1_TextLoader.py`

Loads a plain `.txt` file and inspects the resulting `Document` object.

```python
from pathlib import Path
from langchain_community.document_loaders import TextLoader

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "TheFrenchRevolution.txt"

loader = TextLoader(str(DATA_FILE), encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Content preview: {documents[0].page_content[:200]}")
print(f"Metadata: {documents[0].metadata}")
```

- `TextLoader` treats the entire file as a single `Document` — `len(documents)` will be 1.
- `documents[0].metadata` will contain `{"source": "/path/to/TheFrenchRevolution.txt"}`.

**Run it:**
```bash
python "examples/DocumentLoading&Processing/1_TextLoader.py"
```

---

### Example 2 — PDF Loader

**File:** `examples/DocumentLoading&Processing/2_PDFLoader.py`

Loads a PDF and splits it automatically — one `Document` per page.

```python
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

PDF_FILE = Path(__file__).resolve().parents[1] / "data" / "TheFrenchRevolution.pdf"

loader = PyPDFLoader(str(PDF_FILE))
pages = loader.load()

print(f"Total pages: {len(pages)}")
for i, page in enumerate(pages[:3]):
    print(f"\n--- Page {i+1} ---")
    print(f"Content: {page.page_content[:150]}...")
    print(f"Metadata: {page.metadata}")
```

- `PyPDFLoader` returns one `Document` per page, so `len(pages)` equals the page count of the PDF.
- Each page's metadata includes `{"source": "...", "page": 0}` — the `page` key is zero-indexed.

**Run it:**
```bash
python "examples/DocumentLoading&Processing/2_PDFLoader.py"
```

---

### Example 3 — Web URL Loader

**File:** `examples/DocumentLoading&Processing/3_WebUrlLoader.py`

Fetches and parses web pages directly from a list of URLs.

```python
from langchain_community.document_loaders import WebBaseLoader

URLS = [
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/concepts/",
]

loader = WebBaseLoader(URLS)
docs = loader.load()

print(f"Loaded {len(docs)} web page(s)")
print(f"Content preview: {docs[0].page_content[:200]}...")
print(f"Metadata: {docs[0].metadata}")
```

- Pass a list of URLs; `WebBaseLoader` fetches and strips HTML for each one in a single call.
- Metadata includes the page title and source URL automatically.

**Run it:**
```bash
python "examples/DocumentLoading&Processing/3_WebUrlLoader.py"
```

---

### Example 4 — Directory Loader

**File:** `examples/DocumentLoading&Processing/4_DirectoryLoader.py`

Bulk-loads all files matching a glob pattern from a directory, with optional multithreading.

```python
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

loader = DirectoryLoader(
    DATA_DIR,
    glob="*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True,
)
docs = loader.load()

print(f"Loaded {len(docs)} documents from directory")
```

- `glob` controls which files are matched — swap `"*.txt"` for `"*.pdf"` to load PDFs instead.
- `use_multithreading=True` parallelizes file I/O, which makes a meaningful difference for large directories.

**Run it:**
```bash
python "examples/DocumentLoading&Processing/4_DirectoryLoader.py"
```

---

### Example 5 — Multi-Format Loader Pipeline

**File:** `examples/DocumentLoading&Processing/5_CompleteFileLoaderExample.py`

Combines PDF and text loaders into a single reusable function that ingests a mixed directory in one call.

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

def load_documents(source_dir) -> list:
    all_docs = []

    pdf_loader = DirectoryLoader(
        source_dir, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True
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
```

- `loader_kwargs` passes extra constructor arguments (like `encoding`) through to the underlying loader class.
- The returned list is ready to pass directly into any text splitter in Track 2.

**Run it:**
```bash
python "examples/DocumentLoading&Processing/5_CompleteFileLoaderExample.py"
```

---

## Track 2 — Text Chunking Strategies (Examples 1–4)

### Theory

A **text splitter** divides large documents into smaller chunks that fit within a retrieval context window and carry enough coherence to be useful in isolation. The two key parameters are:

- **`chunk_size`** — the maximum number of characters (or tokens) per chunk.
- **`chunk_overlap`** — the number of characters shared between consecutive chunks, which preserves context across boundaries and prevents answers from being cut in half.

There are three main splitting approaches:

**`CharacterTextSplitter` — fixed separator:**

```python
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
)
```

**`RecursiveCharacterTextSplitter` — cascading separators (recommended):**

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],  # tries each in order
)
```

The recursive splitter tries the first separator; if the resulting chunk is still too large, it falls back to the next one. This produces more natural splits that respect paragraph and sentence boundaries compared to splitting at fixed character positions.

**Why the choice of splitter matters:**

The same document split with different strategies produces chunks of very different character:

| Strategy | Splits On | Best For |
|----------|-----------|----------|
| `CharacterTextSplitter` | A single fixed separator | Clean, regularly-structured text |
| `RecursiveCharacterTextSplitter` | Cascading separators (paragraphs → lines → words) | General prose and mixed documents |
| `TokenTextSplitter` | OpenAI token count via `tiktoken` | When you need hard token budget limits |
| `SemanticChunker` | Embedding-distance jumps between sentences | High-coherence chunks regardless of length |

The `chunk_size` and `chunk_overlap` values are not one-size-fits-all. The right values depend on your retrieval use case — see Example 2 for presets tuned to the most common scenarios.

---

### Example 1 — Three Core Splitting Strategies

**File:** `examples/TextChunkingStrategies/1_Chunking.py`

Compares character-based, recursive, and token-based splitting on the same input text so you can see how chunk counts and boundaries differ.

```python
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

char_splitter      = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
token_splitter     = TokenTextSplitter(chunk_size=50, chunk_overlap=10)

print("Character-based chunks:", len(char_splitter.split_text(SAMPLE_TEXT)))
print("Recursive chunks:",       len(recursive_splitter.split_text(SAMPLE_TEXT)))
print("Token-based chunks:",     len(token_splitter.split_text(SAMPLE_TEXT)))
```

- `TokenTextSplitter` counts OpenAI-style tokens via `tiktoken` — useful when you need hard token budget limits rather than character limits.
- For most use cases, `RecursiveCharacterTextSplitter` produces the best results because it respects natural text boundaries.

**Run it:**
```bash
python "examples/TextChunkingStrategies/1_Chunking.py"
```

---

### Example 2 — Use-Case-Tuned Chunk Parameters

**File:** `examples/TextChunkingStrategies/2_OptimalChunkingParameters.py`

Demonstrates that optimal `chunk_size` and `chunk_overlap` values depend on the retrieval use case, and provides a factory function with four presets.

```python
def create_optimized_splitter(use_case: str):
    configs = {
        "general":      {"chunk_size": 1000, "chunk_overlap": 200},
        "code":         {"chunk_size": 800,  "chunk_overlap": 100,
                         "separators": ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "]},
        "qa":           {"chunk_size": 500,  "chunk_overlap": 50},
        "long_context": {"chunk_size": 2000, "chunk_overlap": 400},
    }
    config = configs.get(use_case, configs["general"])
    return RecursiveCharacterTextSplitter(**config)

splitter = create_optimized_splitter("qa")
chunks = splitter.split_documents(documents)
```

- Smaller `chunk_size` (e.g., 500 for Q&A) keeps retrieved passages tightly focused, reducing noise in the context.
- Larger `chunk_size` (e.g., 2000 for long-context) preserves narrative flow across sections where individual sentences lack enough context.
- The `"code"` preset uses class and function definition boundaries as primary separators, keeping logical units intact.

**Run it:**
```bash
python "examples/TextChunkingStrategies/2_OptimalChunkingParameters.py"
```

---

### Example 3 — Semantic Chunking

**File:** `examples/TextChunkingStrategies/3_AdvancedSemanticChunking.py`

Uses embedding similarity to find natural topic boundaries rather than splitting at fixed character counts.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings

semantic_splitter = SemanticChunker(
    AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    ),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

semantic_chunks = semantic_splitter.split_documents(documents)
print(f"Semantic chunks: {len(semantic_chunks)}")
```

- `SemanticChunker` embeds every sentence, then splits where the cosine distance between adjacent sentences exceeds a threshold.
- `breakpoint_threshold_type="percentile"` with `amount=95` splits at the top 5% of distance jumps — the points where topic changes are sharpest.
- **Trade-off:** Semantic chunking requires an embeddings API call per sentence, making it slower and costlier than character-based approaches. Use it when chunk coherence is more important than cost.

**Run it:**
```bash
python "examples/TextChunkingStrategies/3_AdvancedSemanticChunking.py"
```

---

### Example 4 — End-to-End Chunk Pipeline

**File:** `examples/TextChunkingStrategies/4_PracticalChunkingExample.py`

Wraps load → split → inspect into a single `process_documents()` function ready to drop into any RAG pipeline.

```python
def process_documents(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
```

- Use `.split_documents()` (not `.split_text()`) — it preserves the `Document` metadata (source path, page number) on every resulting chunk, which is essential for source citations in Track 4.

**Run it:**
```bash
python "examples/TextChunkingStrategies/4_PracticalChunkingExample.py"
```

---

## Track 3 — Vector Stores and Similarity Search (Examples 1–2)

### Theory

A **vector store** indexes document chunks as dense embedding vectors and serves approximate nearest-neighbor queries at runtime. Given a query, it embeds the question with the same model used during indexing, then returns the `k` chunks whose embeddings are geometrically closest.

LangChain uses ChromaDB as the local vector store in these examples. Chroma persists its index to disk, so you only pay the embedding cost once — subsequent queries load the existing store from disk without re-embedding.

**Step-by-step: what happens when you index a document**

```
Raw text chunks
        ↓  AzureOpenAIEmbeddings
List of float vectors (one per chunk)
        ↓  Chroma.from_documents()
Persistent index on disk (chroma_db/)
        ↓  vectorstore.as_retriever()
Retriever ready to answer queries
```

**Step-by-step: what happens at query time**

```
User question (string)
        ↓  AzureOpenAIEmbeddings (same model as ingestion)
Query vector
        ↓  cosine similarity against index
Top-k nearest chunk vectors
        ↓  look up original text
List[Document] returned to the RAG chain
```

**Three search modes compared:**

```
similarity_search(query, k)
        ↓  cosine similarity
Top-k most similar chunks (by embedding distance)

similarity_search_with_score(query, k)
        ↓  cosine similarity + distance value
Top-k chunks, each paired with a float score (lower = more similar in Chroma)

max_marginal_relevance_search(query, k, fetch_k, lambda_mult)
        ↓  relevance + diversity
k chunks that balance closeness to the query with diversity among results
```

The `lambda_mult` parameter controls the relevance–diversity tradeoff in MMR:
- `lambda_mult=1.0` → maximum relevance (identical to similarity search)
- `lambda_mult=0.0` → maximum diversity (ignores similarity score entirely)
- `lambda_mult=0.5` → balanced (recommended default)

---

### Example 1 — ChromaDB Setup and Sample Search

**File:** `examples/VectorStores&SimilaritySearch/1_ChromaDBSetup&SampleSearch.py`

Creates a persistent Chroma vector store from a text file and demonstrates all three search modes.

```python
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR),
    collection_name="the_french_revolution",
)

# Similarity search
results = vectorstore.similarity_search("What caused the French Revolution?", k=3)

# With relevance scores
results_with_scores = vectorstore.similarity_search_with_score(
    "What happened during the Reign of Terror?", k=3
)

# MMR for diverse results
results_mmr = vectorstore.max_marginal_relevance_search(
    "Reign of Terror", k=4, fetch_k=20, lambda_mult=0.5
)
```

- `Chroma.from_documents()` creates a new store and persists it to `CHROMA_DIR` in a single call.
- On subsequent runs, load the existing store with `Chroma(persist_directory=..., embedding_function=...)` to avoid paying the embedding cost again.

**Run it:**
```bash
python "examples/VectorStores&SimilaritySearch/1_ChromaDBSetup&SampleSearch.py"
```

---

### Example 2 — DocumentVectorStore Class

**File:** `examples/VectorStores&SimilaritySearch/2_CompleteVectorStoreExample.py`

Wraps Chroma into a reusable `DocumentVectorStore` class with `create_from_documents`, `load_existing`, `search`, and `add_documents` methods.

```python
class DocumentVectorStore:
    def __init__(self, persist_directory: str = _CHROMA_DIR):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        )
        self.vectorstore: Chroma | None = None

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        ).split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        return self.vectorstore

    def search(self, query: str, k: int = 3, method: str = "similarity"):
        if method == "mmr":
            return self.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=k*5)
        return self.vectorstore.similarity_search(query, k=k)
```

- `method="similarity_score"` returns `(Document, float)` tuples, useful for filtering results below a confidence threshold.
- `add_documents()` lets you grow an existing index without recreating it from scratch.

**Run it:**
```bash
python "examples/VectorStores&SimilaritySearch/2_CompleteVectorStoreExample.py"
```

---

## Track 4 — Building RAG Pipelines (Examples 1–6)

### Theory

A **RAG chain** connects a retriever to a language model: the retriever fetches relevant context for a given question, and the LLM uses that context to generate a grounded answer.

Without RAG, every LLM answer draws solely from training data:

```
# Without RAG
User: What caused the French Revolution?
AI:   The French Revolution was caused by a variety of factors...  ← generic, possibly hallucinated
```

With RAG, the answer is anchored to your actual documents:

```
# With RAG
User: What caused the French Revolution?
AI:   According to the document, the main causes were...  ← grounded in retrieved text
```

#### How a modern RAG pipeline works

The pipeline has four stages:

1. **Ingestion** — documents are loaded, split, embedded, and stored in a vector store (done once, offline).
2. **Retrieval** — at query time, the question is embedded and the top-k nearest chunks are fetched from the store.
3. **Augmentation** — retrieved chunks are injected into a prompt template as `{context}`.
4. **Generation** — the LLM generates an answer conditioned on the retrieved context, not its training data.

```
User question
        ↓
Embed query with the same model used at ingestion
        ↓
Retrieve top-k chunks from vector store
        ↓
Inject chunks into prompt template as {context}
        ↓
LLM generates grounded answer
        ↓
Return answer (+ optional source metadata)
```

#### LCEL RAG chains

The modern way to build RAG chains is with **LCEL (LangChain Expression Language)** — the same pipe-operator syntax used across LangChain 1.x. The fundamental pattern is:

```
retriever | format_docs → context
                               ↘
                                prompt | llm | StrOutputParser
                               ↗
RunnablePassthrough → question
```

`StrOutputParser` extracts the plain text string from the model's `AIMessage` response, so the final output of the chain is a `str` rather than a message object. `RunnablePassthrough` forwards the raw question unchanged into the prompt's `{question}` slot without any transformation.

#### Conversational RAG

Standard RAG is stateless — each question is answered independently. Without conversation memory:

```
# Stateless RAG
User: What caused the French Revolution?
AI:   Financial crisis, social inequality, and weak leadership.
User: Can you elaborate on the second cause?
AI:   I'm not sure what you're referring to.   ← no memory of prior answer
```

Conversational RAG adds a **history-aware retriever** that reformulates follow-up questions as standalone queries before retrieval:

```
# Conversational RAG
User: What caused the French Revolution?
AI:   Financial crisis, social inequality, and weak leadership.
User: Can you elaborate on the second cause?
      ↓ history-aware retriever rewrites to:
      "Elaborate on social inequality as a cause of the French Revolution."
AI:   Social inequality in 18th-century France...   ← correctly grounded
```

#### Conversation history

A **conversation history** is a list of `HumanMessage` and `AIMessage` objects representing the full exchange so far. Unlike a black-box memory class, it is a plain Python list — you can inspect, slice, or serialize it at any point.

```python
chat_history = []

result = rag_chain.invoke({"input": "What caused the Revolution?", "chat_history": chat_history})
chat_history.extend([
    HumanMessage(content="What caused the Revolution?"),
    AIMessage(content=result["answer"]),
])
```

For multi-user deployments, maintain a separate `chat_history` list per user session, keyed by a session ID in a dict or database:

```python
histories = {}

def get_history(session_id: str) -> list:
    if session_id not in histories:
        histories[session_id] = []
    return histories[session_id]
```

#### RAG chain components reference

| Component | Purpose |
|-----------|---------|
| `create_retrieval_chain` | Classic chain: retriever → stuff documents → LLM |
| `create_stuff_documents_chain` | Formats retrieved docs into a `{context}` string for the prompt |
| `RunnablePassthrough` | Forwards the raw question unchanged through an LCEL chain |
| `StrOutputParser` | Extracts the plain text string from a model's `AIMessage` response |
| `MultiQueryRetriever` | Generates multiple search queries from one question to broaden recall |
| `create_history_aware_retriever` | Rewrites follow-up questions using chat history before retrieval |
| `MessagesPlaceholder` | Reserves a slot in the prompt for injected conversation history |

---

### Example 1 — Basic RAG Chain

**File:** `examples/BuildingRAGPipelines/1_BasicRAGChain.py`

The simplest complete RAG pipeline: load → chunk → embed → retrieve → answer.

```python
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep the answer concise.\n\nContext: {context}"
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

- `result["answer"]` contains the generated response string.
- `result["context"]` contains the list of retrieved `Document` objects — useful for displaying sources.

**Run it:**
```bash
python "examples/BuildingRAGPipelines/1_BasicRAGChain.py"
```

---

### Example 2 — Modern RAG with LCEL

**File:** `examples/BuildingRAGPipelines/2_ModernRAGwithLCEL.py`

Rebuilds the RAG chain using the LCEL pipe syntax for a more composable, inspectable structure.

```python
from langchain_core.runnables import RunnablePassthrough

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

response = rag_chain.invoke("What is machine learning?")
print(response)
```

- The dict `{"context": retriever | format_docs, "question": RunnablePassthrough()}` is itself a `Runnable` that fetches context and forwards the question simultaneously before passing both to the prompt.
- The final output is a plain `str` rather than a dict — simpler for display, though you lose direct access to the source documents that `create_retrieval_chain` preserves in `result["context"]`.

**Run it:**
```bash
python "examples/BuildingRAGPipelines/2_ModernRAGwithLCEL.py"
```

---

### Example 3 — RAG with Source Citations

**File:** `examples/BuildingRAGPipelines/3_RAGwithSourceCitations.py`

Extends the LCEL chain with a custom document formatter that injects source path and page number into the context block, instructing the model to cite them in its answer.

```python
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Document {i+1}] (Source: {source}, Page: {page})\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

template = """Answer the question based on the following context.
After your answer, list the sources you used with their page numbers.

Context: {context}
Question: {question}

Answer format:
[Your detailed answer]

Sources:
- [Source 1 with page number]
"""
```

- Metadata preservation from `.split_documents()` (not `.split_text()`) is what makes per-chunk citations possible — the source and page number are carried on every chunk as it flows through the pipeline.

**Run it:**
```bash
python "examples/BuildingRAGPipelines/3_RAGwithSourceCitations.py"
```

---

### Example 4 — Multi-Query Retrieval

**File:** `examples/BuildingRAGPipelines/4_MultiQueryRAG.py`

Uses `MultiQueryRetriever` to generate several rephrased versions of the user's question, run them all against the vector store, and deduplicate the combined results — broadening recall for complex or ambiguous queries.

```python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# One question → multiple internal searches → deduplicated results
unique_docs = multi_query_retriever.invoke(
    "What were the main causes of the French Revolution?"
)
print(f"Retrieved {len(unique_docs)} unique documents")
```

- The LLM generates alternative phrasings of the query automatically — no manual work required.
- `unique_docs` is deduplicated, so the same chunk is never returned twice regardless of how many sub-queries matched it.

**Run it:**
```bash
python "examples/BuildingRAGPipelines/4_MultiQueryRAG.py"
```

---

### Example 5 — Conversational RAG

**File:** `examples/BuildingRAGPipelines/5_ConversationalRAG.py`

Adds conversation memory to RAG using `create_history_aware_retriever`. Follow-up questions that reference prior answers are rewritten into standalone queries before retrieval.

```python
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate the question as a standalone query using the chat history. "
               "Do NOT answer — just reformulate if needed, otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []
for query in DEMO_QUERIES:
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=result["answer"]),
    ])
```

- `chat_history` is a plain list of `HumanMessage` / `AIMessage` objects — no special memory class required.
- The contextualization step fires a secondary LLM call only when history exists, so the first turn has no overhead.

**Production tip:** For multi-user deployments, maintain a separate `chat_history` list per `session_id`, keyed in a dict or database.

**Run it:**
```bash
python "examples/BuildingRAGPipelines/5_ConversationalRAG.py"
```

---

### Example 6 — Production-Ready RAG System

**File:** `examples/BuildingRAGPipelines/6_CompleteProductionReadyRAGSystem.py`

Packages the full RAG stack into a configurable `ProductionRAGSystem` class. Features:

- MMR retrieval with a configurable diversity–relevance tradeoff
- Streaming output via `StreamingStdOutCallbackHandler`
- `RunnableParallel` for concurrent source retrieval and answer generation
- `batch_query()` for processing multiple questions in sequence

```python
class ProductionRAGSystem:
    def __init__(self, persist_directory, temperature=0.0, k=4, streaming=True):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=temperature,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )
        self.chain = self._build_chain()

    def query(self, question: str) -> dict:
        result = self.chain.invoke(question)
        return {
            "answer": result["answer"],
            "source_documents": [...],
            "num_sources": len(result["source_documents"]),
        }
```

**Run it:**
```bash
python "examples/BuildingRAGPipelines/6_CompleteProductionReadyRAGSystem.py"
```

---

### Capstone — PDF Q&A Chatbot

**File:** `examples/CompleteRAGImplementation.py`

Brings every concept together into a production-quality `PDFChatbot` class. Features:

- Multi-threaded PDF ingestion from an entire directory
- Persistent ChromaDB vector store that skips re-indexing on restart
- History-aware conversational retrieval with MMR diversity
- Interactive REPL with `reset` (clear conversation history) and `quit` commands

```python
class PDFChatbot:
    def __init__(self, pdf_directory, persist_directory,
                 chunk_size=1000, chunk_overlap=200, k=4):
        self.embeddings = AzureOpenAIEmbeddings(...)
        self.llm = AzureChatOpenAI(...)
        self.chat_history: List[BaseMessage] = []

    def initialize(self):
        documents = self.load_pdfs()
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)
        self.setup_chain()

    def ask(self, question: str) -> dict:
        result = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
        })
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=result["answer"]),
        ])
        return {"answer": result["answer"], "sources": [...]}
```

**Run it:**
```bash
python "examples/CompleteRAGImplementation.py"
```

Type a question and press Enter. Type `reset` to clear conversation history, or `quit` to exit.

---

## Core Concepts at a Glance

| Concept | What It Does | Where to Start |
|---------|-------------|----------------|
| **Document Loaders** | Ingest text, PDFs, and web pages into `Document` objects | Track 1, Example 1 |
| **Metadata** | Source path and page number carried on every `Document` chunk | Track 1, Example 2 |
| **Text Splitters** | Divide documents into retrieval-friendly chunks | Track 2, Example 1 |
| **Embeddings** | Convert text to dense vectors for semantic search | Track 3, Example 1 |
| **ChromaDB** | Local persistent vector store for indexing and retrieval | Track 3, Example 1 |
| **MMR Search** | Retrieves diverse, non-redundant results from a vector store | Track 3, Example 1 |
| **RAG Chain** | Retriever + prompt + LLM assembled into a grounded Q&A pipeline | Track 4, Example 1 |
| **LCEL RAG** | Pipe-style RAG chain composition using the `\|` operator | Track 4, Example 2 |
| **Source Citations** | Per-chunk metadata surfaced as document references in answers | Track 4, Example 3 |
| **MultiQueryRetriever** | Broadens recall by generating multiple query variants automatically | Track 4, Example 4 |
| **Conversational RAG** | History-aware retrieval that handles follow-up questions correctly | Track 4, Examples 5–6 |

---

## Navigating the Examples

**If you're brand new to RAG:** Start at Track 1, Example 1 and work through the tracks in order — each script is self-contained and introduces exactly one new concept.

**If you know the loading and chunking basics and want to build your first pipeline:** Jump to Track 4, Example 1 for the classic retrieval chain, then try Example 2 for the cleaner LCEL version.

**If you need source citations or conversational follow-ups:** Go directly to Example 3 for citations and Example 5 for history-aware retrieval — both extend the base LCEL chain with minimal additions.

**If you want a production reference:** Go straight to `examples/CompleteRAGImplementation.py`. It combines everything — multi-format ingestion, persistent indexing, MMR retrieval, conversational memory, and error handling — into a single deployable chatbot class.

**Running any example:**
```bash
python "examples/<track-folder>/<filename>.py"
```

All examples that use embeddings or the LLM load credentials from `keys/.env` automatically.

---

## Retrieval Strategy Comparison

| Strategy | How It Works | Best For | Example |
|----------|-------------|----------|---------|
| **Similarity Search** | Returns top-k chunks by cosine similarity | Straightforward factual queries | Track 3, Ex 1 |
| **Similarity with Scores** | Same as above, plus distance values for debugging | Evaluating retrieval quality | Track 3, Ex 1 |
| **MMR** | Balances relevance and diversity across results | Broad questions where redundancy is a problem | Track 3, Ex 1 |
| **Multi-Query** | Generates multiple phrasings, deduplicates results | Ambiguous or multi-faceted questions | Track 4, Ex 4 |
| **History-Aware** | Rewrites follow-ups as standalone queries before retrieval | Multi-turn conversational Q&A | Track 4, Ex 5–6 |

---

## Migrating from Legacy APIs

This repository targets LangChain 1.x. If you have existing code written against LangChain 0.x, the two most common patterns have moved:

### Old approach (`ConversationBufferMemory` + `ConversationalRetrievalChain`)

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
result = chain({"question": "What caused the Revolution?"})
```

### New approach (explicit `chat_history` list)

```python
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []
result = rag_chain.invoke({"input": "What caused the Revolution?", "chat_history": chat_history})
chat_history.extend([HumanMessage(content=query), AIMessage(content=result["answer"])])
```

The new approach makes history management explicit and testable — there is no hidden state inside a memory object. `create_history_aware_retriever` replaces `ConversationalRetrievalChain` entirely, and the same `chat_history` list works with any LCEL-based retrieval chain.

---

## Additional Resources

- [LangChain 1.x RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Core Concepts Reference](https://python.langchain.com/docs/concepts/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [LangChain Community Discord](https://discord.gg/langchain)

---

<p align="center">
  Built for learning. Designed for production readiness.<br>
  Licensed under Apache 2.0.
</p>
