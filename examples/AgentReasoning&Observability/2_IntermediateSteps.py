# =============================================================================
# Section 4.2 — Capturing Intermediate Steps
# Topic:  In LangChain 1.0, intermediate steps are accessed via result["messages"].
#         Each AIMessage with tool_calls is a reasoning step; each ToolMessage
#         is the observation. This replaces return_intermediate_steps=True.
#
# The messages list enables:
#   - Post-hoc auditing of agent decisions
#   - Custom step-level logging or metrics
#   - Building UIs that show reasoning progress
# =============================================================================

import os
from pathlib import Path
from typing import Any

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
        "Search for information about the company, including employees and revenue."
    ),
)

tools = [calculator, retriever_tool]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)


def extract_steps(messages: list) -> list[dict[str, Any]]:
    """Extract (tool, input, output) step tuples from the message list."""
    steps = []
    pending_tool_calls: dict[str, dict] = {}

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                pending_tool_calls[tc["id"]] = {"tool": tc["name"], "input": tc["args"]}

        if type(msg).__name__ == "ToolMessage":
            call_id = getattr(msg, "tool_call_id", None)
            if call_id and call_id in pending_tool_calls:
                step = pending_tool_calls.pop(call_id)
                step["output"] = msg.content
                steps.append(step)

    return steps


if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUERY}]})

    # Access the reasoning chain
    print("Final Answer:", result["messages"][-1].content)

    print("\nReasoning Steps:")
    for step in extract_steps(result["messages"]):
        print(f"\nAction: {step['tool']}")
        print(f"Input: {step['input']}")
        print(f"Output: {step['output']}")
