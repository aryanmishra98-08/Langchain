# =============================================================================
# Section 6.1 — Production Issue 1: Over-invocation of Tools
# Topic:  Agents that call the same tool repeatedly with identical inputs.
#
# Symptoms:
#   Tool: search  → Observation: $5M
#   Tool: search  → Observation: $5M   (same — no new information)
#   Tool: search  → Observation: $5M   (same again)
#
# Three solutions are demonstrated:
#   1. Better tool description — add "IMPORTANT: Only call this once per query"
#   2. DeduplicationMiddleware — intercepts repeated (tool, input) pairs in
#      before_model and raises ValueError to stop the loop
#   3. System prompt rules — explicit instruction not to repeat tool calls
#
# All three can be combined; the middleware is the most reliable safety net
# because it enforces the constraint in code rather than relying on the LLM.
# =============================================================================

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
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_QUERY      = "What is the company revenue?"  # demo query
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# ── Solution 1: Improve tool descriptions ─────────────────────────────────────

@tool
def search_once(query: str) -> str:
    """Search for information. IMPORTANT: Only call this once per unique query.
    If you've already searched for this information, use the previous result."""
    results = {
        "company revenue": "Annual revenue is $5 million.",
        "employees": "We have 50 employees.",
    }
    return results.get(query.lower(), f"No results found for: {query}")


tools = [search_once]


# ── Solution 2: DeduplicationMiddleware ──────────────────────────────────────
# Middleware wraps the full agent loop, making it easier to track state across
# the entire invocation (vs callbacks which fire per-event).

class DeduplicationMiddleware(BaseMiddleware):
    def __init__(self):
        self.seen_calls: list[str] = []

    def before_model(self, state, config):
        """Inspect pending tool calls and raise if duplicated."""
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            for tc in getattr(last, "tool_calls", []):
                signature = f"{tc['name']}:{tc['args']}"
                if signature in self.seen_calls:
                    raise ValueError(f"Duplicate tool call detected: {signature}")
                self.seen_calls.append(signature)
        return state, config


# ── Solution 3: System prompt reinforcement ───────────────────────────────────

anti_loop_prompt = (
    "You are a helpful assistant. "
    "IMPORTANT: Never call the same tool with the same input more than once. "
    "If a tool returns no results, accept that and answer with what you know."
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=anti_loop_prompt,
    middleware=[DeduplicationMiddleware()],
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": DEMO_QUERY}]})
    print(result["messages"][-1].content)
