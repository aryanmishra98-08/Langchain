# =============================================================================
# Section 6.1 — Production Debugging: Issue 1 — Over-invocation of Tools
# Topic:  Agents that call the same tool repeatedly with similar inputs.
#
# Symptoms:
#   Action: search  → Observation: $5M
#   Action: search  → Observation: $5M   (same again)
#   Action: search  → Observation: $5M   (same again)
#
# Solutions demonstrated:
#   1. Hard iteration limit via max_iterations
#   2. DeduplicationCallback — raises on repeated (tool, input) pairs
#   3. Improved tool description with explicit "call once" instruction
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
MAX_ITERATIONS  = 5                          # hard cap on tool calls
DEMO_QUERY      = "What is the company revenue?"  # demo query
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# ── Solution 3: Improve tool descriptions ────────────────────────────────────

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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)


# ── Solution 2: DeduplicationCallback ────────────────────────────────────────

class DeduplicationCallback(BaseCallbackHandler):
    def __init__(self):
        self.seen_actions = []

    def on_agent_action(self, action, **kwargs):
        action_signature = f"{action.tool}:{action.tool_input}"

        if action_signature in self.seen_actions:
            raise ValueError(f"Duplicate action detected: {action_signature}")

        self.seen_actions.append(action_signature)


# ── Solution 1: Limit iterations ─────────────────────────────────────────────

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=MAX_ITERATIONS,  # Hard limit
    verbose=True,
)

if __name__ == "__main__":
    dedup_callback = DeduplicationCallback()
    result = agent_executor.invoke(
        {"input": DEMO_QUERY},
        config={"callbacks": [dedup_callback]},
    )
    print(result["output"])
