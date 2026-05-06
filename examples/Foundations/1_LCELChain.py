# =============================================================================
# Section 1.1 — Agent vs Chain: Understanding the Difference
# Topic:  Simple deterministic chain using LCEL (LangChain Expression Language)
# =============================================================================
# A chain has a fixed, pre-defined sequence of operations.
# The pipe operator (|) composes Runnables into a chain and is the modern
# replacement for the deprecated LLMChain class.
#
# Migration note: LLMChain is deprecated and scheduled for removal.
# LCEL (the | pipe operator) is now the standard for composing chains.
# It provides better streaming, batching, async support, and observability.
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_TEXT       = "Long text..."             # input text to summarize
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)

# Fixed sequence: prompt → LLM → output parser
# The pipe operator (|) composes Runnables into a chain
chain = (
    PromptTemplate.from_template("Summarize: {text}")
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    result = chain.invoke({"text": DEMO_TEXT})  # Always follows same path
    print(result)
