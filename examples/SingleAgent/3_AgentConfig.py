# =============================================================================
# Section 3.3 — Agent Configuration
# Topic:  Key AgentExecutor parameters with inline explanations.
#         Use this file as a configuration reference when building agents.
#
# Parameter summary:
#   verbose                 → Print step-by-step reasoning to stdout
#   handle_parsing_errors   → Gracefully retry on LLM output parse failures
#   max_iterations          → Hard cap to prevent infinite tool-call loops
#   max_execution_time      → Wall-clock timeout in seconds
#   early_stopping_method   → "force" is the only reliably supported option;
#                             returns a "Stopped" message when limit is hit
#   return_intermediate_steps → Include full reasoning trace in the output dict
#
# Note on early_stopping_method: "generate" requires the agent class to
# implement a special return method and is not supported by all agent types.
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_tool_calling_agent, AgentExecutor

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE    = 0           # 0 = deterministic output
API_VERSION        = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
MAX_ITERATIONS     = 10          # hard cap on reasoning steps
MAX_EXECUTION_TIME = 60          # wall-clock timeout in seconds
EARLY_STOPPING     = "force"     # "force" returns a Stopped message at the limit
DEMO_QUERY         = "What is 42 * 100?"  # demo calculation query
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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# All key AgentExecutor parameters demonstrated
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                              # Show agent reasoning
    handle_parsing_errors=True,                # Gracefully handle LLM output errors
    max_iterations=MAX_ITERATIONS,             # Prevent infinite loops
    max_execution_time=MAX_EXECUTION_TIME,     # Timeout in seconds
    early_stopping_method=EARLY_STOPPING,      # "force" returns "Stopped" message
    return_intermediate_steps=True,            # Get full reasoning trace in output
)

if __name__ == "__main__":
    result = agent_executor.invoke({"input": DEMO_QUERY})
    print("Output:", result["output"])

    print("\nIntermediate Steps:")
    for step in result.get("intermediate_steps", []):
        action, observation = step
        print(f"  Tool: {action.tool}")
        print(f"  Input: {action.tool_input}")
        print(f"  Output: {observation}")
