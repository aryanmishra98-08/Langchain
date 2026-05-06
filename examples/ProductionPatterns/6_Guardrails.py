# =============================================================================
# Section 6.3 — Guardrails and Constraints
# Topic:  GuardrailAgent wraps an AgentExecutor with pre-execution input
#         checks and post-execution output checks. Queries that violate rules
#         are rejected before the agent runs; outputs containing PII are
#         filtered before being returned.
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_tool_calling_agent, AgentExecutor

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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
base_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ── GuardrailAgent ────────────────────────────────────────────────────────────

class GuardrailAgent:
    """Agent with safety guardrails."""

    def __init__(self, executor: AgentExecutor, rules: Dict):
        self.executor = executor
        self.rules = rules

    def invoke(self, input_dict: Dict) -> Dict:
        query = input_dict["input"]

        # Pre-execution checks
        if not self._check_input_safety(query):
            return {
                "output": "Query rejected: violates safety rules",
                "error": "SAFETY_VIOLATION",
            }

        # Execute with monitoring
        try:
            result = self.executor.invoke(input_dict)

            # Post-execution validation
            if not self._check_output_safety(result["output"]):
                return {
                    "output": "Output filtered: contains prohibited content",
                    "error": "OUTPUT_FILTERED",
                }

            return result

        except Exception as e:
            return {
                "output": f"Execution failed: {str(e)}",
                "error": "EXECUTION_ERROR",
            }

    def _check_input_safety(self, query: str) -> bool:
        """Check if input is safe."""
        # Check query length
        if len(query) > self.rules.get("max_query_length", 1000):
            return False

        # Check for prohibited terms
        prohibited = self.rules.get("prohibited_terms", [])
        if any(term.lower() in query.lower() for term in prohibited):
            return False

        return True

    def _check_output_safety(self, output: str) -> bool:
        """Check if output is safe."""
        # Email pattern
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", output):
            if not self.rules.get("allow_pii", False):
                return False

        # Phone pattern
        if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", output):
            if not self.rules.get("allow_pii", False):
                return False

        return True


# ── Usage ─────────────────────────────────────────────────────────────────────

rules = {
    "max_query_length": MAX_QUERY_LENGTH,
    "prohibited_terms": PROHIBITED_TERMS,
    "allow_pii": ALLOW_PII,
}

guarded_agent = GuardrailAgent(base_executor, rules)

if __name__ == "__main__":
    # Safe query
    result = guarded_agent.invoke({"input": DEMO_QUERY_SAFE})
    print("Safe query result:", result)

    # Unsafe query — contains a prohibited term
    result = guarded_agent.invoke({"input": DEMO_QUERY_UNSAFE})
    print("Unsafe query result:", result)
