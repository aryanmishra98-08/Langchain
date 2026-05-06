# =============================================================================
# Section 5.3 — Multi-Agent Pattern 3: Critic + Builder
# Topic:  An iterative loop where a Builder LCEL chain generates code from
#         requirements and a Critic LCEL chain reviews it. The loop continues
#         until the Critic responds "APPROVED: <reason>" or max_iterations
#         is reached.
# =============================================================================
# Both Builder and Critic are LCEL chains (prompt | llm | StrOutputParser).
# No AgentExecutor is needed — pure chain orchestration is sufficient here.
#
# Critique format expected from Critic:
#   "APPROVED: <reason>"      → loop terminates
#   "NEEDS_WORK: <issues>"    → builder is asked to revise
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE   = 0.7          # creative temperature for builder LLM
API_VERSION       = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
MAX_ITERATIONS    = 3            # max build-critique loop iterations
DEMO_REQUIREMENTS = """Create a Python function that:
1. Takes a list of numbers
2. Removes duplicates
3. Sorts in descending order
4. Returns only even numbers
Include error handling."""
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# ── Builder Chain (LCEL) ──────────────────────────────────────────────────────

builder_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a code builder. Create Python code based on requirements.
If you receive feedback, improve your code accordingly.
Return ONLY the code, no explanations."""),
    ("human", "{input}"),
])

builder_chain = builder_prompt | llm | StrOutputParser()


# ── Critic Chain (LCEL) ───────────────────────────────────────────────────────

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a code critic. Review code for:
1. Correctness
2. Efficiency
3. Best practices
4. Edge cases
5. Security issues

If code is good, respond: "APPROVED: <reason>"
If code needs improvement, respond: "NEEDS_WORK: <specific issues>" """),
    ("human", "Review this code:\n\n{code}"),
])

critic_chain = critic_prompt | llm | StrOutputParser()


# ── Orchestration ─────────────────────────────────────────────────────────────

def critic_builder_loop(requirements: str, max_iterations: int = 3) -> dict:
    """Iterative building and critique."""

    print("=" * 80)
    print("STARTING CRITIC-BUILDER LOOP")
    print("=" * 80)
    print(f"Requirements: {requirements}\n")

    code = None
    feedback_history = []

    for iteration in range(max_iterations):
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'=' * 80}")

        # Build
        print("\n🔨 BUILDER:")
        if iteration == 0:
            builder_input = requirements
        else:
            builder_input = f"""{requirements}

Previous attempt:
{code}

Feedback:
{feedback_history[-1]}

Improve the code based on the feedback."""

        # LCEL chain with StrOutputParser returns string directly
        code = builder_chain.invoke({"input": builder_input})
        print(code)

        # Critique
        print("\n🔍 CRITIC:")
        critique = critic_chain.invoke({"code": code})
        print(critique)

        feedback_history.append(critique)

        # Check if approved
        if "APPROVED" in critique:
            print(f"\n✅ Code approved after {iteration + 1} iteration(s)!")
            return {
                "code": code,
                "iterations": iteration + 1,
                "feedback_history": feedback_history,
                "status": "approved",
            }

    print(f"\n⚠️ Max iterations reached. Returning last version.")
    return {
        "code": code,
        "iterations": max_iterations,
        "feedback_history": feedback_history,
        "status": "max_iterations_reached",
    }


if __name__ == "__main__":
    result = critic_builder_loop(DEMO_REQUIREMENTS, max_iterations=MAX_ITERATIONS)

    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print("\nFinal Code:")
    print(result["code"])
