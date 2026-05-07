# =============================================================================
# Section 6.5 — Monitoring and Observability
# Topic:  AgentMonitor (metrics collector) + MonitoringMiddleware (lifecycle
#         hooks) for tracking call counts, success rates, execution times,
#         tool usage frequencies, and recent errors.
# =============================================================================
# AgentMonitor.get_report() returns:
#   total_calls          — number of agent invocations
#   success_rate         — successful / total
#   average_execution_time — mean wall-clock time per call
#   tool_usage           — {tool_name: invocation_count}
#   recent_errors        — last 5 error records with timestamp and message
#
# MonitoringMiddleware wires AgentMonitor into the agent loop:
#   before_model → log_start() on first call
#   after_tool   → log_tool_use(tool_name)
#   after_model  → detect finish and log_success() or log_failure()
#
# LangChain 1.0: middleware replaces MonitoringCallback on AgentExecutor.
# Callbacks still work but middleware provides cleaner state access.
# =============================================================================

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_agent
from langchain.agents.middleware import BaseMiddleware

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


# ── AgentMonitor ──────────────────────────────────────────────────────────────

class AgentMonitor:
    """Comprehensive agent monitoring."""

    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "tool_usage": {},
            "execution_times": [],
            "errors": [],
        }
        self.start_time: float | None = None

    def log_start(self):
        self.start_time = time.time()
        self.metrics["total_calls"] += 1

    def log_tool_use(self, tool_name: str):
        self.metrics["tool_usage"][tool_name] = (
            self.metrics["tool_usage"].get(tool_name, 0) + 1
        )

    def log_success(self):
        if self.start_time is None:
            return
        self.metrics["successful_calls"] += 1
        self.metrics["execution_times"].append(time.time() - self.start_time)

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
        times = self.metrics["execution_times"]
        return {
            "total_calls": self.metrics["total_calls"],
            "success_rate": (
                self.metrics["successful_calls"] / self.metrics["total_calls"]
                if self.metrics["total_calls"] > 0 else 0
            ),
            "average_execution_time": sum(times) / len(times) if times else 0,
            "tool_usage": self.metrics["tool_usage"],
            "recent_errors": self.metrics["errors"][-5:],
        }


# ── MonitoringMiddleware ───────────────────────────────────────────────────────

class MonitoringMiddleware(BaseMiddleware):
    def __init__(self, monitor: AgentMonitor):
        self.monitor = monitor
        self._started = False

    def before_model(self, state, config):
        if not self._started:
            self.monitor.log_start()
            self._started = True
        return state, config

    def after_tool(self, state, config):
        messages = state.get("messages", [])
        tool_messages = [m for m in messages if type(m).__name__ == "ToolMessage"]
        if tool_messages:
            tool_name = getattr(tool_messages[-1], "name", "unknown")
            self.monitor.log_tool_use(tool_name)
        return state, config

    def after_model(self, state, config):
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            # Final answer: AIMessage with content and no tool_calls
            if type(last).__name__ == "AIMessage" and last.content and not getattr(last, "tool_calls", None):
                self.monitor.log_success()
                self._started = False
        return state, config

    def on_error(self, error, state, config):
        self.monitor.log_failure(str(error))
        self._started = False
        raise error


# ── Usage ─────────────────────────────────────────────────────────────────────

monitor = AgentMonitor()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
    middleware=[MonitoringMiddleware(monitor)],
)

if __name__ == "__main__":
    for q in DEMO_QUERIES:
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"Q: {q}  →  A: {result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("MONITOR REPORT:")
    print(json.dumps(monitor.get_report(), indent=2))
