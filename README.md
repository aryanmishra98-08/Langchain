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
├── requirements.txt
├── keys/                              # API key storage
├── myenv/                             # Python 3.11 virtual environment
├── examples/
│   ├── data/
│   │   └── TheFrenchRevolution.txt    # Sample dataset
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

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key (create a .env file or export directly)
export OPENAI_API_KEY="your-api-key-here"
```

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
- **OpenAI**: `text-embedding-ada-002` (1536 dims)
- **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **Cohere**: `embed-english-v3.0`

**Cost vs Performance:**
- Larger dimensions = Better accuracy, Higher cost
- Smaller dimensions = Faster search, Lower accuracy

---

## 3. Document Loading & Processing

> **Examples:** `examples/DocumentLoading&Processing/`

### 3.1 Setup & Installation

```bash
# Core dependencies
pip install langchain langchain-community langchain-openai
pip install chromadb tiktoken

# Document loaders
pip install pypdf python-docx unstructured
pip install beautifulsoup4 lxml

# Additional utilities
pip install python-dotenv
```

### 3.2 Loading Different Document Types

#### 3.2.1 Text Files

> See: `examples/DocumentLoading&Processing/1_TextLoader.py`

```python
from langchain_community.document_loaders import TextLoader

# Load a single text file
loader = TextLoader("data/document.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Content preview: {documents[0].page_content[:200]}")
print(f"Metadata: {documents[0].metadata}")
```

#### 3.2.2 PDF Documents

> See: `examples/DocumentLoading&Processing/2_PDFLoader.py`

```python
from langchain_community.document_loaders import PyPDFLoader

# Load PDF with page-level granularity
loader = PyPDFLoader("data/research_paper.pdf")
pages = loader.load()

print(f"Total pages: {len(pages)}")
for i, page in enumerate(pages[:3]):
    print(f"\n--- Page {i+1} ---")
    print(f"Content: {page.page_content[:150]}...")
    print(f"Metadata: {page.metadata}")
```

#### 3.2.3 Web Pages

> See: `examples/DocumentLoading&Processing/3_WebUrlLoader.py`

```python
from langchain_community.document_loaders import WebBaseLoader

# Load from URL
loader = WebBaseLoader([
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/concepts/",
])
docs = loader.load()

print(f"Loaded {len(docs)} web page(s)")
```

#### 3.2.4 Directory Loader (Multiple Files)

> See: `examples/DocumentLoading&Processing/4_DirectoryLoader.py`

```python
from langchain_community.document_loaders import DirectoryLoader

# Load all text files from directory
loader = DirectoryLoader(
    "data/documents/",
    glob="**/*.txt",
    show_progress=True,
    use_multithreading=True
)
docs = loader.load()

print(f"Loaded {len(docs)} documents from directory")
```

### 3.3 Complete Loading Example

> See: `examples/DocumentLoading&Processing/5_CompleteFileLoaderExample.py`

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

load_dotenv()

def load_documents(source_dir: str) -> list:
    """
    Load documents from multiple sources
    """
    all_docs = []
    
    # Load PDFs
    pdf_loader = DirectoryLoader(
        source_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    pdf_docs = pdf_loader.load()
    all_docs.extend(pdf_docs)
    
    # Load text files
    txt_loader = DirectoryLoader(
        source_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    txt_docs = txt_loader.load()
    all_docs.extend(txt_docs)
    
    print(f"✓ Loaded {len(pdf_docs)} PDFs")
    print(f"✓ Loaded {len(txt_docs)} text files")
    print(f"✓ Total documents: {len(all_docs)}")
    
    return all_docs

# Usage
documents = load_documents("./data")
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

> See: `examples/TextChunkingStrategies/1_Chunking.py`

```python
from langchain.text_splitter import (
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

> See: `examples/TextChunkingStrategies/2_OptimalChunkingParameters.py`

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

> See: `examples/TextChunkingStrategies/3_AdvancedSemanticChunking.py`

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Split based on semantic similarity
semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95,
)

semantic_chunks = semantic_splitter.split_documents(documents)
print(f"Semantic chunks: {len(semantic_chunks)}")
```

### 4.5 Practical Chunking Example

> See: `examples/TextChunkingStrategies/4_PracticalChunkingExample.py`

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def process_documents(file_path: str):
    """
    Complete document processing pipeline
    """
    # 1. Load document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # 3. Inspect chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk.page_content)}")
        print(f"Preview: {chunk.page_content[:150]}...")
        print(f"Metadata: {chunk.metadata}")
    
    return chunks

# Usage
chunks = process_documents("data/document.pdf")
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

> See: `examples/VectorStores&SimilaritySearch/1_ChromaDBSetup&SampleSearch.py`

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_documents"
)

print(f"✓ Created vector store with {vectorstore._collection.count()} documents")
```

### 5.3 Similarity Search

```python
# Basic similarity search
query = "What is machine learning?"
results = vectorstore.similarity_search(
    query,
    k=3  # Return top 3 most similar documents
)

print(f"Found {len(results)} relevant documents:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}\n")
```

### 5.4 Similarity Search with Scores

```python
# Get similarity scores
query = "Explain neural networks"
results_with_scores = vectorstore.similarity_search_with_score(
    query,
    k=3
)

print("Results with similarity scores:\n")
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

> See: `examples/VectorStores&SimilaritySearch/2_CompleteVectorStoreExample.py`

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

class DocumentVectorStore:
    """
    Manages document vectorization and retrieval
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
    
    def create_from_documents(self, documents: list):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✓ Vector store created with {len(chunks)} chunks")
        return self.vectorstore
    
    def load_existing(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vectorstore
    
    def search(self, query: str, k: int = 3, method: str = "similarity"):
        if method == "similarity":
            return self.vectorstore.similarity_search(query, k=k)
        elif method == "mmr":
            return self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=k*5
            )
        elif method == "similarity_score":
            return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def add_documents(self, documents: list):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        print(f"✓ Added {len(chunks)} new chunks")

# Usage
if __name__ == "__main__":
    loader = PyPDFLoader("data/document.pdf")
    docs = loader.load()
    
    vs = DocumentVectorStore()
    vs.create_from_documents(docs)
    
    results = vs.search("What are the main findings?", k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:200])
```

---

## 6. Building RAG Pipelines

> **Examples:** `examples/BuildingRAGPipelines/`

### 6.1 Basic RAG Chain (Legacy)

> See: `examples/BuildingRAGPipelines/1_BasicRAGChain.py`

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff", "map_reduce", "refine", "map_rerank"
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# Query
result = qa_chain({"query": "What is the main topic of the document?"})
print("Answer:", result["result"])
```

### 6.2 Modern RAG with LCEL (Recommended)

> See: `examples/BuildingRAGPipelines/2_ModernRAGwithLCEL.py`

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
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

# Build RAG chain using LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is machine learning?")
print(response)
```

### 6.3 RAG with Source Citations

> See: `examples/BuildingRAGPipelines/3_RAGwithSourceCitations.py`

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
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
    """Format documents with source information"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        formatted.append(
            f"[Document {i+1}] (Source: {source}, Page: {page})\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

rag_chain_with_sources = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain_with_sources.invoke("Explain the key concepts")
print(response)
```

### 6.4 Multi-Query RAG (Advanced)

> See: `examples/BuildingRAGPipelines/4_MultiQueryRAG.py`

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Multi-query retriever generates multiple search queries
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# Single user query generates multiple searches
question = "What are the benefits of machine learning?"
unique_docs = multi_query_retriever.get_relevant_documents(query=question)

print(f"Retrieved {len(unique_docs)} unique documents")
```

### 6.5 Conversational RAG (With Memory)

> See: `examples/BuildingRAGPipelines/5_ConversationalRAG.py`

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True
)

# Multi-turn conversation
queries = [
    "What is machine learning?",
    "What are its main applications?",
    "Can you explain more about the first application?"
]

for query in queries:
    print(f"\nUser: {query}")
    result = conversational_chain({"question": query})
    print(f"Assistant: {result['answer']}")
```

### 6.6 Complete Production-Ready RAG System

> See: `examples/BuildingRAGPipelines/6_CompleteProductionReadyRAGSystem.py`

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List, Dict
import os

class ProductionRAGSystem:
    """
    Production-ready RAG system with best practices
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model: str = "gpt-4",
        temperature: float = 0,
        k: int = 4
    ):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3}
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
        
        return (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> Dict[str, any]:
        relevant_docs = self.retriever.get_relevant_documents(question)
        answer = self.chain.invoke(question)
        
        return {
            "answer": answer,
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ],
            "num_sources": len(relevant_docs)
        }
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        return [self.query(q) for q in questions]

# Usage
if __name__ == "__main__":
    rag = ProductionRAGSystem(persist_directory="./chroma_db", model="gpt-4", k=3)
    result = rag.query("What is the main topic discussed?")
    
    print("\n\nAnswer:", result["answer"])
    print(f"\nUsed {result['num_sources']} sources")
```

---

## 7. Advanced RAG Patterns

### 7.1 Retriever Comparison

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 1. Similarity search
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 2. MMR (Maximum Marginal Relevance) - Diverse results
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

# 3. Similarity with threshold
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 3}
)

# Compare results
query = "machine learning applications"

print("Similarity Search:")
sim_docs = similarity_retriever.get_relevant_documents(query)
print(f"Found {len(sim_docs)} documents\n")

print("MMR Search (Diverse):")
mmr_docs = mmr_retriever.get_relevant_documents(query)
print(f"Found {len(mmr_docs)} documents\n")

print("Threshold Search:")
threshold_docs = threshold_retriever.get_relevant_documents(query)
print(f"Found {len(threshold_docs)} documents")
```

### 7.2 Contextual Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Compressor extracts only relevant parts
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# Compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Query
query = "What are the applications of deep learning?"
compressed_docs = compression_retriever.get_relevant_documents(query)

print(f"Retrieved {len(compressed_docs)} compressed documents")
for doc in compressed_docs:
    print(f"\n{doc.page_content}")
```

### 7.3 Parent Document Retriever

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Store for parent documents
store = InMemoryStore()

# Small chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

# Larger chunks for context
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Parent document retriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents
parent_retriever.add_documents(documents)

# Query - retrieves small chunks but returns large context
docs = parent_retriever.get_relevant_documents(
    "What is neural network architecture?"
)
```

### 7.4 Self-Query Retriever

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

# Define metadata fields
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source document name",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="The page number in the source document",
        type="integer"
    ),
]

# Document content description
document_content_description = "Technical documentation about AI and ML"

# Self-query retriever
llm = ChatOpenAI(model="gpt-4", temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# Natural language query with filters
query = "What does the document say about neural networks from page 5?"
docs = retriever.get_relevant_documents(query)
```

### 7.5 Ensemble Retriever

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Vector retriever (semantic search)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# BM25 retriever (keyword search)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# Ensemble combines both
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Equal weight
)

# Query
docs = ensemble_retriever.get_relevant_documents(
    "machine learning algorithms"
)
print(f"Ensemble retrieved {len(docs)} documents")
```

---

## 8. Intermediate Project: PDF Q&A Chatbot

> **Full implementation:** `examples/CompleteRAGImplementation.py`

### 8.1 Project Overview

**Goal:** Build a chatbot that can answer questions about uploaded PDF documents

**Features:**
- Multiple PDF upload
- Persistent vector store
- Conversational memory
- Source citation
- Streaming responses

### 8.2 Complete Implementation

```python
# pdf_chatbot.py
"""
PDF Q&A Chatbot with LangChain
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


class PDFChatbot:
    """
    Chatbot for answering questions about PDF documents
    """
    
    def __init__(
        self,
        pdf_directory: str,
        persist_directory: str = "./pdf_chatbot_db",
        model: str = "gpt-4",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4
    ):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.vectorstore = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.chain = None
    
    def load_pdfs(self) -> List:
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} pages from PDFs")
        return documents
    
    def split_documents(self, documents: List) -> List:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List):
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        print(f"✓ Vector store ready with {self.vectorstore._collection.count()} embeddings")
    
    def setup_chain(self):
        system_template = """You are a helpful AI assistant that answers questions about PDF documents.

Use the following context to answer the user's question. If you don't know the answer
or can't find it in the context, say so clearly.

Context:
{context}

Chat History:
{chat_history}

Instructions:
1. Answer based on the provided context
2. Be specific and cite relevant information
3. If the answer isn't in the context, say "I cannot find this information in the provided documents"
4. Maintain conversation continuity using chat history

Answer the question thoughtfully and accurately."""
        
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.k * 3}
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_template(system_template)
            }
        )
    
    def initialize(self):
        documents = self.load_pdfs()
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)
        self.setup_chain()
        print("\n✅ Chatbot is ready! Start asking questions.\n")
    
    def ask(self, question: str) -> Dict:
        result = self.chain({"question": question})
        return {
            "answer": result["answer"],
            "sources": [
                {
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "content_preview": doc.page_content[:150] + "..."
                }
                for doc in result["source_documents"]
            ]
        }
    
    def chat(self):
        print("=" * 60)
        print("PDF Q&A CHATBOT")
        print("=" * 60)
        print("Ask questions about your PDF documents!")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("Type 'reset' to clear conversation history")
        print("=" * 60 + "\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            if question.lower() == 'reset':
                self.memory.clear()
                print("\n🔄 Conversation history cleared!\n")
                continue
            if not question:
                continue
            
            try:
                result = self.ask(question)
                print(f"\n🤖 Assistant: {result['answer']}")
                if result['sources']:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"   {i}. {source['source']} (Page {source['page']})")
                print()
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


# Main execution
if __name__ == "__main__":
    PDF_DIRECTORY = "./data/pdfs"
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    
    chatbot = PDFChatbot(
        pdf_directory=PDF_DIRECTORY,
        persist_directory="./pdf_chatbot_db",
        model="gpt-4",
        chunk_size=1000,
        chunk_overlap=200,
        k=3
    )
    chatbot.initialize()
    chatbot.chat()
```

### 8.3 Usage Instructions

```bash
# 1. Install dependencies
pip install langchain langchain-community langchain-openai
pip install chromadb pypdf tiktoken python-dotenv

# 2. Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 3. Create data directory and add PDFs
mkdir -p data/pdfs
# Copy your PDF files to data/pdfs/

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

You: What is the main topic of the research paper?

🤖 Assistant: The main topic of the research paper is "Deep Learning for
Natural Language Processing." The paper explores various neural network
architectures for NLP tasks, including transformers, BERT, and GPT models.

📚 Sources:
   1. research_paper.pdf (Page 1)
   2. research_paper.pdf (Page 2)

You: What are the key findings?

🤖 Assistant: The key findings from the research include:
1. Transformer models outperform RNNs by 23% on machine translation tasks
2. Pre-training on large corpora significantly improves downstream task performance
3. Attention mechanisms allow models to capture long-range dependencies effectively

📚 Sources:
   1. research_paper.pdf (Page 8)
   2. research_paper.pdf (Page 12)

You: quit

👋 Goodbye!
```

---

## 9. Performance Optimization

### 9.1 Chunk Size Optimization

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

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

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

def compare_embedding_models(texts):
    """Compare different embedding models"""
    models = {
        "OpenAI": OpenAIEmbeddings(),
        "HuggingFace (MiniLM)": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        "HuggingFace (MPNet)": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    }
    
    results = {}
    for name, model in models.items():
        start = time.time()
        embeddings = model.embed_documents(texts[:100])
        elapsed = time.time() - start
        
        results[name] = {
            "time": elapsed,
            "dimension": len(embeddings[0]),
            "cost": "Paid" if "OpenAI" in name else "Free"
        }
    
    print("\nEmbedding Model Comparison:")
    print(f"{'Model':<30} {'Time (s)':<12} {'Dimensions':<12} {'Cost':<10}")
    print("-" * 70)
    for name, data in results.items():
        print(f"{name:<30} {data['time']:<12.3f} {data['dimension']:<12} {data['cost']:<10}")
    
    return results

sample_texts = [chunk.page_content for chunk in chunks[:100]]
compare_embedding_models(sample_texts)
```

### 9.3 Retrieval Optimization

```python
def optimize_retrieval_params(vectorstore, test_queries, k_values=[3, 5, 7, 10]):
    """Test different k values for retrieval"""
    print("\nRetrieval Parameter Optimization:")
    print(f"{'k Value':<10} {'Avg Docs':<12} {'Avg Relevance':<15}")
    print("-" * 40)
    
    for k in k_values:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.5}
        )
        
        total_docs = 0
        total_relevance = 0
        
        for query in test_queries:
            docs = retriever.get_relevant_documents(query)
            total_docs += len(docs)
            total_relevance += len(docs) * 0.8
        
        avg_docs = total_docs / len(test_queries)
        avg_relevance = total_relevance / len(test_queries)
        
        print(f"{k:<10} {avg_docs:<12.1f} {avg_relevance:<15.2f}")

test_queries = [
    "What is machine learning?",
    "Explain neural networks",
    "What are the applications?"
]
optimize_retrieval_params(vectorstore, test_queries)
```

### 9.4 Caching Strategies

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
import time

# Enable caching
set_llm_cache(InMemoryCache())

llm = ChatOpenAI(model="gpt-4")

# First call - no cache
start = time.time()
response1 = llm.invoke("What is 2+2?")
time1 = time.time() - start

# Second call - cached
start = time.time()
response2 = llm.invoke("What is 2+2?")
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

**Task:** Implement and compare three different text splitters

```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)

# Your code here:
# 1. Load a document
# 2. Split using three different splitters
# 3. Compare the number and quality of chunks
# 4. Determine which is best for your use case
```

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
from langchain.retrievers import EnsembleRetriever
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
    ("What is ML?", "Machine learning is a type of AI..."),
    ("Define neural networks", "Neural networks are...")
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
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Concepts](https://python.langchain.com/docs/concepts/#retrieval-augmented-generation)
- [Vector Stores](https://python.langchain.com/docs/integrations/vectorstores/)
- [Retrievers](https://python.langchain.com/docs/concepts/#retrievers)

### Best Practices
1. **Chunking:** Start with 1000 chars, 200 overlap
2. **Retrieval:** Use k=3-5 for most queries
3. **Embeddings:** OpenAI for quality, HuggingFace for cost
4. **Vector Store:** Chroma for dev, Pinecone for production
5. **Prompt Engineering:** Always instruct to cite sources

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

**Happy Learning! 🚀**
