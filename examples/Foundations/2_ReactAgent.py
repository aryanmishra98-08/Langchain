# =============================================================================
# Section 1.1 — Agent vs Chain: Understanding the Difference
# Section 1.3 — Tool Selection Logic
# Topic:  Basic ReAct agent that dynamically selects between a search tool and
#         a calculator tool based on the user's query.
# =============================================================================
# An agent has dynamic reasoning and non-deterministic control flow.
# It can choose which tools to invoke based on tool descriptions.
#
# Agent decision process:
#   1. Parse user question
#   2. Match intent to tool descriptions
#   3. Select the most relevant tool
#   4. Execute and observe result
#   5. Continue or terminate
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
MAX_ITERATIONS  = 5                          # max agent reasoning steps before forced stop
DEMO_QUERY      = "What's the population of NYC times 2?"  # demo question
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


def search_function(query: str) -> str:
    """Simulated web search."""
    results = {
        "population of NYC": "NYC has 8.3 million people",
    }
    return results.get(query.lower(), f"Search results for: {query}")


def calculator_function(expression: str) -> str:
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error: {e}"


# Tool descriptions are critical — the agent uses them to decide which tool to use
search_tool = Tool(
    name="Search",
    func=search_function,
    description=(
        "Useful for finding current information about events, people, or facts. "
        "Input should be a search query."
    ),
)

calculator_tool = Tool(
    name="Calculator",
    func=calculator_function,
    description=(
        "Useful for mathematical calculations. "
        "Input should be a mathematical expression like '2 + 2' or '15 * 67'."
    ),
)

tools = [search_tool, calculator_tool]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=MAX_ITERATIONS,
)

if __name__ == "__main__":
    # Agent decides: search for population → calculator for multiplication
    result = executor.invoke({"input": DEMO_QUERY})
    print(result["output"])
