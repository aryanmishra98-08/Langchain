# =============================================================================
# Section 2.2 — Creating Tools: Method 1 — @tool decorator
# Section 2.3 — Best Practices: Clear Descriptions, Error Handling,
#               Consistent Return Format
# Topic:  The @tool decorator pattern with annotated good/bad examples.
# =============================================================================
# The @tool decorator turns a regular Python function into a LangChain tool.
# The docstring becomes the agent's description for tool selection — a vague
# docstring leads to poor tool selection. Type hints define the input schema.
#
# Why numexpr instead of eval()? eval() has known escape vectors that allow
# arbitrary code execution even with restricted __builtins__. numexpr parses
# through a restricted grammar limited to mathematical operations, making it
# the safe choice for calculator tools.
# =============================================================================

from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
DEMO_EXPRESSION = "sqrt(16) * 3"                           # math expression to test
DEMO_USER_ID    = "550e8400-e29b-41d4-a716-446655440000"   # UUID for profile lookup demo
DEMO_CITY       = "New York"                                # city for weather lookup demo
DEMO_CITY_BAD   = "Paris"                                   # city that triggers not-found path
# ──────────────────────────────────────────────────────────────────────────────


# ── ❌ Bad: vague description — agent cannot reliably select this tool ────────

@tool
def get_data(id: str) -> str:
    """Gets data."""
    return f"Data for {id}"


# ── ✅ Good: specific, actionable description ─────────────────────────────────

@tool
def get_user_profile(user_id: str) -> str:
    """Retrieves user profile information including name, email, and registration date.

    Use this when you need detailed information about a specific user.
    Input should be a valid user ID (format: UUID).
    """
    # Simulated fetch
    return f"User {user_id}: name=Alice, email=alice@example.com, registered=2023-01-15"


# ── ✅ Good: error handling (Best Practice #2) ────────────────────────────────

@tool
def safe_calculator(expression: str) -> str:
    """Safely evaluate mathematical expressions.

    Input should be a valid math expression like '2 + 2' or 'sqrt(16) * 3'.
    Supports +, -, *, /, **, sqrt, log, sin, cos, etc.
    """
    try:
        # numexpr is a sandboxed math evaluator — no arbitrary code execution
        result = numexpr.evaluate(expression).item()
        return f"Result: {result}"
    except Exception as e:
        return f"Error: Could not evaluate expression. {str(e)}"


# ── ✅ Good: consistent return format (Best Practice #3) ─────────────────────

@tool
def weather_lookup(city: str) -> str:
    """Get current weather for a city."""
    try:
        # Simulated weather data
        weather_data = {
            "New York": ("72°F", "Sunny"),
            "London": ("55°F", "Cloudy"),
        }
        if city not in weather_data:
            raise KeyError(city)
        temp, conditions = weather_data[city]
        return f"Temperature: {temp}, Conditions: {conditions}"
    except KeyError:
        return f"Error: City '{city}' not found. Please check spelling."
    except Exception as e:
        return f"Error: Weather service unavailable. {str(e)}"


if __name__ == "__main__":
    print(safe_calculator.invoke(DEMO_EXPRESSION))
    print(get_user_profile.invoke(DEMO_USER_ID))
    print(weather_lookup.invoke(DEMO_CITY))
    print(weather_lookup.invoke(DEMO_CITY_BAD))  # Triggers "not found" error path
