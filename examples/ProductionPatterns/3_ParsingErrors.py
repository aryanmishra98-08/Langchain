# =============================================================================
# Section 6.3 — Production Debugging: Issue 3 — Parsing Errors
# Topic:  LLM returns malformed action/input that raises OutputParserException.
#
# Symptoms (ReAct / text-parsed agents only):
#   OutputParserException: Could not parse LLM output:
#   `Action: calculator\nInput: calculate 2+2 please`
#
# Root cause: ReAct agents parse freeform text from the LLM. Any deviation
# from the expected "Action: X\nAction Input: Y" format causes a crash.
#
# Solutions in LangChain 1.0:
#   1. create_agent uses tool-calling APIs (structured JSON) — parsing errors
#      are eliminated entirely. This is the recommended approach.
#   2. For LLMs that don't support tool-calling (open-source models), use
#      LangGraph's prebuilt create_react_agent with handle_parsing_errors
#      (covered in the LangGraph session).
#
# This file shows why create_agent solves the problem by design.
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
DEMO_QUERY      = "What is 42 * 100?"  # query sent to the agent
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


# ── Solution: create_agent (tool-calling, no text parsing) ───────────────────
# The LLM returns structured JSON tool calls — there is no freeform text to
# parse, so OutputParserException cannot occur. This is the current standard.

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUERY}]})
    print("=== create_agent (no parsing errors possible) ===")
    print(result["messages"][-1].content)
