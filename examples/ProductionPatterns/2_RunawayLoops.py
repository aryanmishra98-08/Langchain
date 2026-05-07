# =============================================================================
# Section 6.2 — Production Debugging: Issue 2 — Runaway Loops
# Topic:  Agents that enter an infinite loop without making progress.
#
# Symptoms:
#   Tool: search  → Observation: No results
#   Tool: search  → Observation: No results  (repeats forever)
#
# Solutions demonstrated:
#   1. LoopGuardMiddleware — detects when the same observation repeats
#   2. Enhanced system prompt with explicit loop-prevention rules
#   3. Async timeout wrapper for wall-clock limits
#
# LangChain 1.0: max_execution_time (AgentExecutor param) is replaced by
# wrapping agent.invoke() in asyncio.wait_for() or concurrent.futures.
# =============================================================================

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_agent
from langchain.agents.middleware import BaseMiddleware

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE    = 0           # 0 = deterministic output
API_VERSION        = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
TIMEOUT_SECONDS    = 60          # wall-clock timeout
MAX_NO_PROGRESS    = 2           # repeated observations before LoopGuardMiddleware raises
DEMO_QUERY         = "Find information about unknown_topic_xyz"  # triggers loop
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


@tool
def search(query: str) -> str:
    """Search for information."""
    # Simulated: unknown topics always return empty to demonstrate the loop problem
    if "unknown" in query.lower():
        return "No results found"
    return f"Results for '{query}': Some relevant information found."


tools = [search]


# ── Solution 1: LoopGuardMiddleware ──────────────────────────────────────────
# Tracks tool outputs and raises when the same result appears repeatedly,
# indicating the agent is not making progress.

class LoopGuardMiddleware(BaseMiddleware):
    def __init__(self, max_no_progress: int = 3):
        self.observations: list[str] = []
        self.max_no_progress = max_no_progress

    def after_tool(self, state, config):
        messages = state.get("messages", [])
        tool_messages = [m for m in messages if type(m).__name__ == "ToolMessage"]
        if tool_messages:
            last_output = str(tool_messages[-1].content)
            recent = self.observations[-self.max_no_progress:]
            if recent and all(o == last_output for o in recent):
                raise ValueError("No progress detected — same observation repeated")
            self.observations.append(last_output)
        return state, config


# ── Solution 2: Better prompting with explicit loop-prevention rules ──────────

enhanced_system_prompt = """Answer the question using available tools.

IMPORTANT RULES:
1. If a tool returns "No results", DO NOT retry the same query
2. If you cannot find information after 2 attempts, say "I don't have enough information"
3. Do not loop — each action should make progress toward the answer"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=enhanced_system_prompt,
    middleware=[LoopGuardMiddleware(max_no_progress=MAX_NO_PROGRESS)],
)


# ── Solution 3: Async timeout wrapper ─────────────────────────────────────────

async def invoke_with_timeout(query: str, timeout: int) -> str:
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent.invoke,
                {"messages": [{"role": "user", "content": query}]},
            ),
            timeout=timeout,
        )
        return result["messages"][-1].content
    except asyncio.TimeoutError:
        return f"Agent timed out after {timeout}s"


if __name__ == "__main__":
    output = asyncio.run(invoke_with_timeout(DEMO_QUERY, TIMEOUT_SECONDS))
    print(output)
