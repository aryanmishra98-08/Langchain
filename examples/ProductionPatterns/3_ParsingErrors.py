# =============================================================================
# Section 6.3 — Production Issue 3: Parsing Errors
# Topic:  Why tool-calling eliminates OutputParserException by design.
#
# The problem with text-parsed agents (ReAct): the LLM must produce output
# in an exact format like "Action: X\nAction Input: Y". Any deviation —
# extra text, wrong case, a natural-language preamble — raises:
#   OutputParserException: Could not parse LLM output
#
# create_agent avoids this entirely by using the model's native tool-calling
# API. The LLM returns structured JSON tool calls, so there is no freeform
# text to parse and no OutputParserException possible.
#
# For open-source LLMs without tool-calling support, LangGraph's prebuilt
# create_react_agent with handle_parsing_errors is the fallback (LangGraph
# session covers this).
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
