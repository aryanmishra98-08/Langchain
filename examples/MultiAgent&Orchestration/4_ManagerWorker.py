# =============================================================================
# Section 5.4 — Multi-Agent Communication: Hierarchical Pattern
# Topic:  A ManagerWorkerSystem where a Manager LCEL chain parses a high-level
#         task into subtask assignments and dispatches them to named Worker
#         AgentExecutors.
# =============================================================================
# Communication pattern:
#         Manager
#        /   |   \
#   Worker1 Worker2 Worker3
#
# Manager output format: "ASSIGN: worker_name | task_description"
# The _parse_assignments method extracts these lines and routes subtasks.
#
# To use: instantiate with a list of {"name": str, "executor": AgentExecutor}
# worker dicts, then call system.execute(task).
# =============================================================================

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_tool_calling_agent, AgentExecutor

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
MAX_ITERATIONS  = 5                          # max steps per worker agent
DEMO_TASK       = "Fetch user data from the database and calculate 1000 * 12 for the annual projection."
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# ── Example worker tools ──────────────────────────────────────────────────────

@tool
def fetch_data(source: str) -> str:
    """Fetch data from a specified source."""
    data_map = {
        "database": "User records: 1000 active users",
        "api": "Weather: 72°F, Sunny",
    }
    return data_map.get(source.lower(), f"No data for source: {source}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error: {e}"


def _make_executor(tools_list):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful worker agent. Complete the assigned task."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm=llm, tools=tools_list, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools_list, verbose=True, max_iterations=MAX_ITERATIONS)


# ── ManagerWorkerSystem ───────────────────────────────────────────────────────

class ManagerWorkerSystem:
    def __init__(self, llm, workers: list):
        self.llm = llm
        self.workers = {w["name"]: w["executor"] for w in workers}
        self.manager_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a manager agent coordinating workers.
Available workers: {workers}

For each subtask, assign it to the appropriate worker.
Format: ASSIGN: worker_name | task_description"""),
            ("human", "{task}"),
        ])
        # Build LCEL chain for manager
        self.manager_chain = self.manager_prompt | self.llm

    def execute(self, task: str) -> Dict:
        # Manager delegates
        manager_response = self.manager_chain.invoke({
            "task": task,
            "workers": ", ".join(self.workers.keys()),
        })

        assignments = self._parse_assignments(manager_response.content)

        # Workers execute
        results = {}
        for worker_name, subtask in assignments.items():
            if worker_name not in self.workers:
                results[worker_name] = f"Error: unknown worker '{worker_name}'"
                continue
            print(f"\n🔧 {worker_name} executing: {subtask}")
            result = self.workers[worker_name].invoke({"input": subtask})
            results[worker_name] = result["output"]

        return results

    def _parse_assignments(self, manager_output: str) -> Dict:
        """Parse ASSIGN: worker | task format."""
        assignments = {}
        for line in manager_output.split("\n"):
            if "ASSIGN:" in line:
                parts = line.split("ASSIGN:")[1].split("|")
                if len(parts) == 2:
                    worker = parts[0].strip()
                    task = parts[1].strip()
                    assignments[worker] = task
        return assignments


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_worker = _make_executor([fetch_data])
    calc_worker = _make_executor([calculator])

    system = ManagerWorkerSystem(
        llm=llm,
        workers=[
            {"name": "DataWorker", "executor": data_worker},
            {"name": "CalcWorker", "executor": calc_worker},
        ],
    )

    results = system.execute(DEMO_TASK)

    print("\n" + "=" * 80)
    print("RESULTS:")
    for worker, output in results.items():
        print(f"\n{worker}: {output}")
