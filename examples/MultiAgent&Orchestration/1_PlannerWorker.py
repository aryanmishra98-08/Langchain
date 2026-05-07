# =============================================================================
# Section 5.1 — Multi-Agent Pattern 1: Planner + Worker
# Topic:  One agent (LCEL chain) plans a task as a numbered list; a second
#         agent (create_agent) executes the plan using fetch/process/report
#         tools. Demonstrates sequential (pipeline) communication pattern.
# =============================================================================
# Communication pattern: Agent1 → Result → Agent2
#
# The Planner is an LCEL chain (no agent needed for pure text planning).
# The Worker is a full create_agent with domain tools.
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_agent

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_TASK       = "Create a report on user activity including data from database and API"
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# ── Worker tools ──────────────────────────────────────────────────────────────

@tool
def fetch_data(source: str) -> str:
    """Fetch data from a specified source (database, api, file)."""
    data_map = {
        "database": "User records: 1000 active users, 500 premium",
        "api": "Weather data: 72°F, Sunny",
        "file": "Config: max_users=10000, timeout=30s",
    }
    return data_map.get(source, "Source not found")


@tool
def process_data(data: str) -> str:
    """Process and analyze data."""
    return f"Processed: {len(data)} characters analyzed, summary: {data[:50]}..."


@tool
def generate_report(findings: str) -> str:
    """Generate a formatted report from findings."""
    return f"=== REPORT ===\n{findings}\n=== END REPORT ==="


# ── Planner Chain (LCEL — no agent needed for pure planning) ─────────────────

planner_prompt = PromptTemplate.from_template(
    """You are a planning agent. Your job is to create a step-by-step plan.
Given a task, break it down into discrete steps.
Return ONLY the plan as a numbered list, nothing else.

Task: {input}

Plan:"""
)

planner_chain = planner_prompt | llm | StrOutputParser()


# ── Worker Agent ──────────────────────────────────────────────────────────────

worker_tools = [fetch_data, process_data, generate_report]
worker_agent = create_agent(
    model=llm,
    tools=worker_tools,
    system_prompt="You are a helpful worker agent. Complete the assigned task using the available tools.",
)


# ── Orchestration ─────────────────────────────────────────────────────────────

def planner_worker_system(task: str) -> str:
    """Orchestrate planner and worker agents."""

    print("=" * 80)
    print("STEP 1: PLANNING")
    print("=" * 80)

    # Step 1: Planner creates plan (LCEL chain returns string directly via StrOutputParser)
    plan = planner_chain.invoke({"input": task})
    print(f"\nPlan:\n{plan}\n")

    print("=" * 80)
    print("STEP 2: EXECUTION")
    print("=" * 80)

    # Step 2: Worker executes plan
    worker_task = f"""Execute the following plan:
{plan}

Original Task: {task}

Complete each step and provide the final result."""

    result = worker_agent.invoke({"messages": [{"role": "user", "content": worker_task}]})

    return result["messages"][-1].content


if __name__ == "__main__":
    final_result = planner_worker_system(DEMO_TASK)
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(final_result)
