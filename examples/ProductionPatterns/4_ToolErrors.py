# =============================================================================
# Section 6.4 — Production Issue 4: Tool Execution Errors
# Topic:  Tools that crash (ZeroDivisionError) or stall (API timeouts).
#
# The golden rule for tools: always return a string, never raise.
# When a tool raises, the agent loop crashes instead of recovering gracefully.
#
# Three defensive patterns demonstrated:
#   1. safe_calculator     — catches specific exceptions and returns human-
#                            readable error strings per error type
#   2. validated_calculator — pre-validates the expression before evaluating,
#                            catching obvious errors before numexpr sees them
#   3. run_with_timeout    — cross-platform timeout via concurrent.futures
#                            (works on Windows, Linux, macOS — unlike SIGALRM)
#
# Security note: numexpr's restricted grammar prevents arbitrary code
# execution. Never use eval() for untrusted mathematical input.
# =============================================================================

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
TIMEOUT_SECONDS  = 5             # seconds before timed_operation aborts
DEMO_EXPRESSION  = "10 / 0"      # expression to test zero-division handling
DEMO_EXPRESSION2 = "sqrt(144)"   # valid expression for happy-path test
DEMO_EXPRESSION3 = "2 ** 10"     # second valid expression
DEMO_QUERY       = "langchain agents"  # query for timed_operation demo
# ──────────────────────────────────────────────────────────────────────────────


# ── Solution 1: Specific exception handling ───────────────────────────────────

@tool
def safe_calculator(expression: str) -> str:
    """Perform mathematical calculations safely."""
    try:
        result = numexpr.evaluate(expression).item()
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Cannot divide by zero. Please try a different calculation."
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}. Please check your expression."


# ── Solution 2: Pre-validation ────────────────────────────────────────────────

@tool
def validated_calculator(expression: str) -> str:
    """Calculate with validation."""
    # Pre-validate
    if "/0" in expression.replace(" ", ""):
        return "Error: Division by zero detected"

    # numexpr already blocks dangerous operations, but we add explicit checks
    forbidden = ["import", "__", "exec", "eval", "open", "os.", "sys."]
    if any(d in expression for d in forbidden):
        return "Error: Expression contains forbidden operations"

    try:
        result = numexpr.evaluate(expression).item()
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# ── Solution 3: Cross-platform timeout ───────────────────────────────────────

def run_with_timeout(func, args, seconds: int):
    """Run function with timeout — works on Windows, Linux, and macOS."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=seconds)
        except FuturesTimeoutError:
            return None


def slow_operation(query: str) -> str:
    """Simulates a slow external API call."""
    time.sleep(10)  # Simulate delay
    return f"Results for: {query}"


@tool
def timed_operation(query: str) -> str:
    """Operation with timeout."""
    result = run_with_timeout(slow_operation, (query,), seconds=TIMEOUT_SECONDS)
    if result is None:
        return f"Error: Operation timed out after {TIMEOUT_SECONDS} seconds"
    return result


if __name__ == "__main__":
    print("--- safe_calculator ---")
    print(safe_calculator.invoke(DEMO_EXPRESSION))
    print(safe_calculator.invoke(DEMO_EXPRESSION2))

    print("\n--- validated_calculator ---")
    print(validated_calculator.invoke(DEMO_EXPRESSION))
    print(validated_calculator.invoke(DEMO_EXPRESSION3))

    print(f"\n--- timed_operation (will timeout after {TIMEOUT_SECONDS}s) ---")
    print(timed_operation.invoke(DEMO_QUERY))
