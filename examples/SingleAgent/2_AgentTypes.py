# =============================================================================
# Section 3.2 — Agent Types
# Topic:  Standard agent vs specialist agent using create_agent().
#
# create_agent() uses the model's native tool-calling API (structured JSON),
# which is more reliable than text-parsed approaches. The primary way to
# differentiate agent behavior is via system_prompt — no need to build a
# ChatPromptTemplate manually.
#
# For open-source LLMs that don't support tool calling, LangGraph's prebuilt
# create_react_agent is the fallback (covered in the LangGraph session).
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_agent

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_QUERY      = "What is 15 multiplied by 67?"  # query sent to all agent variants
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
    """Evaluate a mathematical expression like '2 + 2'."""
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error: {e}"


tools = [calculator]


# ── Standard Agent (LangChain 1.0) ───────────────────────────────────────────
# create_agent uses tool-calling APIs internally.
# Default system prompt keeps behavior minimal and focused.

standard_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)


# ── Agent with Custom System Prompt ──────────────────────────────────────────
# Customize behavior via system_prompt — no need to build a ChatPromptTemplate.

specialist_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a math specialist. When given a calculation, always use the "
        "calculator tool and show the expression you evaluated."
    ),
)


if __name__ == "__main__":
    messages = [{"role": "user", "content": DEMO_QUERY}]

    print("=== Standard Agent (create_agent) ===")
    result = standard_agent.invoke({"messages": messages})
    print(result["messages"][-1].content)

    print("\n=== Specialist Agent (custom system_prompt) ===")
    result = specialist_agent.invoke({"messages": messages})
    print(result["messages"][-1].content)
