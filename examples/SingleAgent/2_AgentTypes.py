# =============================================================================
# Section 3.2 — Agent Types in LangChain
# Topic:  Comparison of the three main agent creation patterns:
#           1. ReAct Agent — works with any LLM via prompt-engineered text parsing
#           2. Tool Calling Agent — recommended, uses native tool-calling API
#           3. Structured Chat Agent — for multi-input tools on legacy models
#
# Choosing an agent type:
#   ReAct           → Open-source / non-tool-calling LLMs  (medium reliability)
#   Tool Calling    → Modern OpenAI / Anthropic / Google    (high reliability)
#   Structured Chat → Multi-input tools on legacy models    (medium reliability)
#
# Migration note: create_openai_functions_agent is deprecated.
# Use create_tool_calling_agent — it is model-agnostic (works with OpenAI,
# Anthropic, Google, etc.) and eliminates text-parsing errors.
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain.agents import (
    create_react_agent,
    create_tool_calling_agent,
    create_structured_chat_agent,
    AgentExecutor,
)
from langchain import hub

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_QUERY      = "What is 15 multiplied by 67?"  # query sent to all three agent types
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


# ── 1. ReAct Agent ────────────────────────────────────────────────────────────
# Works with any LLM — uses prompt-engineered text parsing for reasoning.

react_prompt = hub.pull("hwchase17/react")
react_agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


# ── 2. Tool Calling Agent (Recommended) ──────────────────────────────────────
# Uses each provider's native tool-calling API (structured JSON output).
# More reliable than text-parsed ReAct. Replaces deprecated
# create_openai_functions_agent.

tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
tool_calling_agent = create_tool_calling_agent(
    llm=llm, tools=tools, prompt=tool_calling_prompt
)
tool_calling_executor = AgentExecutor(
    agent=tool_calling_agent,
    tools=tools,
    verbose=True,
)


# ── 3. Structured Chat Agent ──────────────────────────────────────────────────
# For multi-input tools on legacy models without native tool-calling support.

structured_prompt = hub.pull("hwchase17/structured-chat-agent")
structured_agent = create_structured_chat_agent(
    llm=llm, tools=tools, prompt=structured_prompt
)
structured_executor = AgentExecutor(
    agent=structured_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


if __name__ == "__main__":
    query = {"input": DEMO_QUERY}

    print("=== 1. ReAct Agent ===")
    print(react_executor.invoke(query)["output"])

    print("\n=== 2. Tool Calling Agent (Recommended) ===")
    print(tool_calling_executor.invoke(query)["output"])

    print("\n=== 3. Structured Chat Agent ===")
    print(structured_executor.invoke(query)["output"])
