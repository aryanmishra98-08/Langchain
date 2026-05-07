# =============================================================================
# Section 4.1 — Understanding Agent Thought Process: Verbose Mode
# Topic:  In LangChain 1.0, the reasoning trace lives in result["messages"].
#         Each message is a typed object: HumanMessage, AIMessage (with
#         tool_calls), ToolMessage (tool result), and the final AIMessage.
#
# This replaces the verbose=True flag on AgentExecutor. You get the same
# Thought → Action → Observation → Final Answer trace by iterating messages.
# =============================================================================
# Expected message sequence:
#
#   HumanMessage  — user's original question
#   AIMessage     — agent reasoning + tool_calls=[{name, args}]
#   ToolMessage   — tool result (observation)
#   AIMessage     — (possibly more tool calls if needed)
#   ...
#   AIMessage     — final answer (no tool_calls, just content)
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
RETRIEVER_K     = 2                          # documents retrieved per query
DEMO_QUERY      = "What is the revenue per employee?"  # demo question
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


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
    system_prompt="You are a helpful assistant. Use the available tools to answer questions.",
)


def print_reasoning_trace(messages: list) -> None:
    """Print the full Thought → Action → Observation → Answer trace."""
    for msg in messages:
        msg_type = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            for tc in tool_calls:
                print(f"\nAction: {tc['name']}")
                print(f"Action Input: {tc['args']}")
        elif msg_type == "ToolMessage":
            print(f"Observation: {msg.content}")
        elif msg_type == "AIMessage" and msg.content:
            print(f"\nThought/Answer: {msg.content}")
        elif msg_type == "HumanMessage":
            print(f"Question: {msg.content}")


if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUERY}]})

    print("=" * 80)
    print("REASONING TRACE")
    print("=" * 80)
    print_reasoning_trace(result["messages"])

    print("\n" + "=" * 80)
    print("Final Answer:", result["messages"][-1].content)
