# =============================================================================
# Section 6.1 — Production Debugging: Issue 2 — Runaway Loops
# Topic:  Agents that enter an infinite loop without making progress.
#
# Symptoms:
#   Thought: I need to find X
#   Action: search  → Observation: No results
#   Thought: I need to find X
#   Action: search  → Observation: No results  (repeats forever)
#
# Solutions demonstrated:
#   1. max_execution_time timeout
#   2. ProgressCallback — detects when the same observation repeats
#   3. Enhanced system prompt with explicit loop-prevention rules
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
LLM_TEMPERATURE    = 0           # 0 = deterministic output
API_VERSION        = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
MAX_EXECUTION_TIME = 60          # wall-clock timeout in seconds
MAX_NO_PROGRESS    = 2           # repeated observations before ProgressCallback raises
DEMO_QUERY         = "Find information about unknown_topic_xyz"  # query that triggers loop
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


# ── Solution 2: Progress tracking callback ───────────────────────────────────

class ProgressCallback(BaseCallbackHandler):
    def __init__(self, max_no_progress: int = 3):
        self.observations = []
        self.max_no_progress = max_no_progress

    def on_tool_end(self, output, **kwargs):
        output_str = str(output)
        # Check if we're getting new information
        if output_str in self.observations[-self.max_no_progress:]:
            raise ValueError("No progress detected - same observation repeated")
        self.observations.append(output_str)


# ── Solution 3: Better prompting with explicit loop-prevention rules ──────────

enhanced_system_prompt = """Answer the question using available tools.

IMPORTANT RULES:
1. If a tool returns "No results", DO NOT retry the same query
2. If you cannot find information after 2 attempts, say "I don't have enough information"
3. Do not loop - each action should make progress toward the answer"""

prompt = ChatPromptTemplate.from_messages([
    ("system", enhanced_system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)


# ── Solution 1: Timeout ───────────────────────────────────────────────────────

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_execution_time=MAX_EXECUTION_TIME,  # timeout in seconds
    verbose=True,
)

if __name__ == "__main__":
    progress_callback = ProgressCallback(max_no_progress=MAX_NO_PROGRESS)
    result = agent_executor.invoke(
        {"input": DEMO_QUERY},
        config={"callbacks": [progress_callback]},
    )
    print(result["output"])
