# =============================================================================
# Section 6.2 — Monitoring and Observability
# Topic:  AgentMonitor (metrics collector) + MonitoringCallback (lifecycle
#         hooks) for tracking call counts, success rates, execution times,
#         tool usage frequencies, and recent errors.
# =============================================================================
# AgentMonitor.get_report() returns:
#   total_calls          — number of times log_start() was called
#   success_rate         — successful / total
#   average_execution_time — mean wall-clock time per call
#   tool_usage           — {tool_name: invocation_count}
#   recent_errors        — last 5 error records with timestamp and message
#
# MonitoringCallback wires the AgentMonitor into the AgentExecutor lifecycle:
#   on_chain_start  → log_start()
#   on_agent_action → log_tool_use(tool)
#   on_agent_finish → log_success()
#   on_chain_error  → log_failure(str(error))
# =============================================================================

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_tool_calling_agent, AgentExecutor

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_QUERIES    = [                           # queries for batch monitoring demo
    "What is 15 * 67?",
    "What is 2 ** 8?",
    "What is sqrt(144)?",
]
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


# ── AgentMonitor ──────────────────────────────────────────────────────────────

class AgentMonitor:
    """Comprehensive agent monitoring."""

    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "tool_usage": {},
            "execution_times": [],
            "errors": [],
        }
        self.start_time = None

    def log_start(self):
        self.start_time = time.time()
        self.metrics["total_calls"] += 1

    def log_tool_use(self, tool_name: str):
        if tool_name not in self.metrics["tool_usage"]:
            self.metrics["tool_usage"][tool_name] = 0
        self.metrics["tool_usage"][tool_name] += 1

    def log_success(self):
        if self.start_time is None:
            return
        execution_time = time.time() - self.start_time
        self.metrics["successful_calls"] += 1
        self.metrics["execution_times"].append(execution_time)

    def log_failure(self, error: str):
        if self.start_time is None:
            return
        execution_time = time.time() - self.start_time
        self.metrics["failed_calls"] += 1
        self.metrics["execution_times"].append(execution_time)
        self.metrics["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "execution_time": execution_time,
        })

    def get_report(self) -> Dict:
        avg_time = (
            sum(self.metrics["execution_times"]) / len(self.metrics["execution_times"])
            if self.metrics["execution_times"] else 0
        )
        return {
            "total_calls": self.metrics["total_calls"],
            "success_rate": (
                self.metrics["successful_calls"] / self.metrics["total_calls"]
                if self.metrics["total_calls"] > 0 else 0
            ),
            "average_execution_time": avg_time,
            "tool_usage": self.metrics["tool_usage"],
            "recent_errors": self.metrics["errors"][-5:],  # Last 5 errors
        }


# ── MonitoringCallback ────────────────────────────────────────────────────────

class MonitoringCallback(BaseCallbackHandler):
    def __init__(self, monitor: AgentMonitor):
        self.monitor = monitor

    def on_chain_start(self, serialized, inputs, **kwargs):
        # Track top-level chain start
        self.monitor.log_start()

    def on_agent_action(self, action, **kwargs):
        self.monitor.log_tool_use(action.tool)

    def on_agent_finish(self, finish, **kwargs):
        self.monitor.log_success()

    def on_chain_error(self, error: BaseException, **kwargs):
        self.monitor.log_failure(str(error))


# ── Usage ─────────────────────────────────────────────────────────────────────

monitor = AgentMonitor()
monitoring_callback = MonitoringCallback(monitor)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[monitoring_callback],
)

if __name__ == "__main__":
    for q in DEMO_QUERIES:
        result = agent_executor.invoke({"input": q})
        print(f"Q: {q}  →  A: {result['output']}")

    print("\n" + "=" * 60)
    print("MONITOR REPORT:")
    print(json.dumps(monitor.get_report(), indent=2))
