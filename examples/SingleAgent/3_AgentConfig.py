# =============================================================================
# Section 3.3 — Agent Configuration in LangChain 1.0
# Topic:  Key create_agent parameters with inline explanations.
#         Use this file as a configuration reference when building agents.
#
# Parameter summary:
#   model           → LLM instance or model string identifier
#   tools           → List of tool functions / Tool objects
#   system_prompt   → Agent instructions (replaces ChatPromptTemplate boilerplate)
#   name            → Identifier used in multi-agent systems
#   middleware      → List of middleware for observability, safety, flow control
#   state_schema    → Custom TypedDict extending AgentState for extra state fields
#   response_format → Constrain output to a specific schema (structured output)
#
# Accessing intermediate steps in 1.0:
#   result["messages"] contains the full conversation including tool calls and
#   tool results as message objects. Iterate to inspect the reasoning trace.
#
# Timeout: wrap agent.invoke() in asyncio or concurrent.futures for wall-clock limits.
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
LLM_TEMPERATURE = 0              # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_QUERY      = "What is 42 * 100?"  # demo calculation query
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
    """Evaluate a mathematical expression."""
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error: {e}"


tools = [calculator]

# create_agent with all key parameters demonstrated
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use tools when needed.",
    name="demo_agent",       # optional; useful for tracing in multi-agent systems
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUERY}]})

    # Final answer is the last message
    print("Output:", result["messages"][-1].content)

    # Inspect the full reasoning trace via the messages list
    print("\nReasoning Trace (all messages):")
    for msg in result["messages"]:
        role = getattr(msg, "type", type(msg).__name__)
        content = msg.content if hasattr(msg, "content") else str(msg)
        # Tool call messages may carry tool_calls metadata instead of content
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"  [{role}] Tool call → {tc['name']}({tc['args']})")
        elif content:
            print(f"  [{role}] {str(content)[:120]}")
