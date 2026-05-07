# =============================================================================
# Section 3.1 — Building Your First Agent
# Topic:  Complete single-agent example with a calculator tool (numexpr) and
#         a document retrieval tool (Chroma). Tests three query types:
#           1. Calculation only
#           2. Retrieval only
#           3. Both tools required
#
# Input:  {"messages": [{"role": "user", "content": "..."}]}
# Output: result["messages"][-1].content
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool, create_retriever_tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.agents import create_agent

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
RETRIEVER_K     = 2                          # number of documents retrieved per query
DEMO_QUERY_1    = "If the company's revenue was $5 million and grew by 30%, what is the new revenue?"
DEMO_QUERY_2    = "When was the company founded?"
DEMO_QUERY_3    = "What is the revenue per employee?"
# ──────────────────────────────────────────────────────────────────────────────

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# Tool 1: Calculator (production-safe via numexpr)
@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations.
    Input should be a valid math expression like '2 + 2' or '(10 * 5) / 2'.
    """
    try:
        result = numexpr.evaluate(expression).item()
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# Tool 2: Document Retrieval
documents = [
    "The company was founded in 2020.",
    "Our main product is an AI assistant.",
    "We have 50 employees across 3 offices.",
    "Annual revenue for 2023 was $5 million.",
]

vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        api_version=API_VERSION,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    ),
)

retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
    name="company_knowledge",
    description=(
        "Search for information about the company, including founding date, "
        "products, employees, and revenue."
    ),
)

tools = [calculator, retriever_tool]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the available tools to answer questions accurately.",
)

if __name__ == "__main__":
    # Query 1: Requires calculation
    result1 = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUERY_1}]})
    print("\n" + "=" * 80)
    print("Result 1:", result1["messages"][-1].content)

    # Query 2: Requires retrieval only
    result2 = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUERY_2}]})
    print("\n" + "=" * 80)
    print("Result 2:", result2["messages"][-1].content)

    # Query 3: Requires both tools
    result3 = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUERY_3}]})
    print("\n" + "=" * 80)
    print("Result 3:", result3["messages"][-1].content)
