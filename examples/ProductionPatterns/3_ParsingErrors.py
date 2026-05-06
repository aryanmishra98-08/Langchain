# =============================================================================
# Section 6.1 — Production Debugging: Issue 3 — Parsing Errors
# Topic:  LLM returns malformed action/input that raises OutputParserException.
#
# Symptoms:
#   OutputParserException: Could not parse LLM output:
#   `Action: calculator\nInput: calculate 2+2 please`
#
# Solutions demonstrated:
#   1. handle_parsing_errors=True — automatic retry on parse failures
#   2. Custom error handler (callable) — returns corrective instruction string
#   3. create_tool_calling_agent — eliminates text-parsing errors entirely
#      (native tool-calling APIs return structured JSON)
#
# Recommendation: Most parsing errors disappear when you switch from
# create_react_agent (text-parsed) to create_tool_calling_agent.
# Use ReAct only when your LLM doesn't support tool calling.
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_react_agent, create_tool_calling_agent, AgentExecutor
from langchain import hub

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_QUERY      = "What is 42 * 100?"  # query sent to all three approaches
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


# ── Solution 1: handle_parsing_errors=True (ReAct agent) ─────────────────────

react_prompt = hub.pull("hwchase17/react")
react_agent = create_react_agent(llm, tools, react_prompt)

agent_executor_v1 = AgentExecutor(
    agent=react_agent,
    tools=tools,
    handle_parsing_errors=True,  # Retry on parse errors
    verbose=True,
)


# ── Solution 2: Custom error handler (string or callable) ────────────────────

def custom_parse_error_handler(error) -> str:
    return "Invalid action format. Use: Action: tool_name\nAction Input: input_value"


agent_executor_v2 = AgentExecutor(
    agent=react_agent,
    tools=tools,
    handle_parsing_errors=custom_parse_error_handler,
)


# ── Solution 3: Tool Calling Agent (Recommended) ─────────────────────────────
# Native tool-calling APIs return structured JSON, eliminating parse errors.

tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
tool_calling_agent = create_tool_calling_agent(llm, tools, tool_calling_prompt)
agent_executor_v3 = AgentExecutor(
    agent=tool_calling_agent,
    tools=tools,
    verbose=True,
)

if __name__ == "__main__":
    query = {"input": DEMO_QUERY}

    print("=== Solution 3: Tool Calling Agent (Recommended) ===")
    print(agent_executor_v3.invoke(query)["output"])
