# =============================================================================
# Section 2.4 — Built-in Tool Examples
# Topic:  Three production-ready tools you can drop into any agent:
#           1. Calculator  — numexpr sandboxed math evaluation (always safe)
#           2. Web Search  — DuckDuckGoSearchRun (requires internet)
#           3. Retrieval   — create_retriever_tool from a Chroma vector store
# =============================================================================
# Security note on PythonREPL: langchain_experimental's PythonREPL executes
# arbitrary Python code. Never expose it to agents that handle untrusted input.
# Use a focused tool like this numexpr-based calculator instead.
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool, create_retriever_tool, Tool
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
API_VERSION      = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_EXPRESSION  = "15 * 67"    # calculator demo expression
DEMO_EXPRESSION2 = "sqrt(144)"  # second calculator demo expression
# ──────────────────────────────────────────────────────────────────────────────


# ── 1. Calculator Tool ────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid math expression like '2 + 2' or '15 * 67'."""
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error: {e}"


# ── 2. Web Search Tool ────────────────────────────────────────────────────────

search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="WebSearch",
    func=search.run,
    description="Search the internet for current information. Input should be a search query.",
)


# ── 3. Retrieval Tool (from Vector Store) ─────────────────────────────────────

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

vectorstore = Chroma.from_texts(
    texts=["Document 1 content", "Document 2 content"],
    embedding=embeddings,
)

retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(),
    name="DocumentRetriever",
    description=(
        "Search through internal documents. "
        "Use this to find information from company knowledge base."
    ),
)


if __name__ == "__main__":
    # Calculator — always safe, no external dependencies
    print(calculator.invoke(DEMO_EXPRESSION))
    print(calculator.invoke(DEMO_EXPRESSION2))

    # Uncomment to test (requires internet / OPENAI_API_KEY):
    # print(search_tool.invoke("LangChain agents overview"))
    # print(retriever_tool.invoke("Document 1"))
