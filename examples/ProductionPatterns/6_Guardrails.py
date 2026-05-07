# =============================================================================
# Section 6.6 — Guardrails and Constraints
# Topic:  GuardrailAgent wraps a create_agent with pre-execution input checks
#         and post-execution output checks. Queries that violate rules are
#         rejected before the agent runs; outputs containing PII are filtered.
# =============================================================================
# Rules dict schema:
#   max_query_length  (int)  — maximum allowed input length
#   prohibited_terms  (list) — case-insensitive blocked input substrings
#   allow_pii         (bool) — if False, outputs containing email addresses
#                              or phone numbers are filtered
#
# Return structure on violation:
#   {"output": "<reason>", "error": "SAFETY_VIOLATION | OUTPUT_FILTERED | EXECUTION_ERROR"}
# =============================================================================

import os
import re
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_agent

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE   = 0            # 0 = deterministic output
API_VERSION       = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
MAX_QUERY_LENGTH  = 500          # maximum allowed query length in characters
PROHIBITED_TERMS  = ["hack", "exploit", "bypass"]  # blocked input substrings (case-insensitive)
ALLOW_PII         = False        # if False, outputs with email/phone are filtered
DEMO_QUERY_SAFE   = "What is 2+2?"           # query that should pass guardrails
DEMO_QUERY_UNSAFE = "How to hack a system?"  # query that should be rejected
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

base_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)


# ── GuardrailAgent ────────────────────────────────────────────────────────────

class GuardrailAgent:
    """Agent with safety guardrails."""

    def __init__(self, agent, rules: Dict):
        self.agent = agent
        self.rules = rules

    def invoke(self, query: str) -> Dict:
        # Pre-execution checks
        if not self._check_input_safety(query):
            return {
                "output": "Query rejected: violates safety rules",
                "error": "SAFETY_VIOLATION",
            }

        # Execute
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            output = result["messages"][-1].content

            # Post-execution validation
            if not self._check_output_safety(output):
                return {
                    "output": "Output filtered: contains prohibited content",
                    "error": "OUTPUT_FILTERED",
                }

            return {"output": output}

        except Exception as e:
            return {
                "output": f"Execution failed: {str(e)}",
                "error": "EXECUTION_ERROR",
            }

    def _check_input_safety(self, query: str) -> bool:
        if len(query) > self.rules.get("max_query_length", 1000):
            return False
        prohibited = self.rules.get("prohibited_terms", [])
        return not any(term.lower() in query.lower() for term in prohibited)

    def _check_output_safety(self, output: str) -> bool:
        if self.rules.get("allow_pii", False):
            return True
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", output):
            return False
        if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", output):
            return False
        return True


# ── Usage ─────────────────────────────────────────────────────────────────────

rules = {
    "max_query_length": MAX_QUERY_LENGTH,
    "prohibited_terms": PROHIBITED_TERMS,
    "allow_pii": ALLOW_PII,
}

guarded_agent = GuardrailAgent(base_agent, rules)

if __name__ == "__main__":
    # Safe query
    result = guarded_agent.invoke(DEMO_QUERY_SAFE)
    print("Safe query result:", result)

    # Unsafe query — contains a prohibited term
    result = guarded_agent.invoke(DEMO_QUERY_UNSAFE)
    print("Unsafe query result:", result)
