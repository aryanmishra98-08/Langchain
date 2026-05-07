# =============================================================================
# Section 2.2 — Creating Tools: Method 2 — Tool class directly
# Topic:  Creating a LangChain tool from a plain Python function using the
#         Tool class constructor.
# =============================================================================
# Use the Tool class when you have an existing function you want to wrap,
# or when you prefer explicit name/description/func separation over decorators.
# The three required fields map directly to how agents use the tool:
#   name        → how the agent refers to it in tool calls
#   func        → the Python callable that runs when the agent invokes it
#   description → what the agent reads to decide whether to use it
# =============================================================================

from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import Tool

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
DEMO_QUERY_NAME  = "Alice"           # search term to test name match
DEMO_QUERY_EMAIL = "bob@example.com" # search term to test email match
# ──────────────────────────────────────────────────────────────────────────────


def search_database(query: str) -> str:
    """Simulate a database query for user records."""
    # Simulated database records
    records = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
    ]
    results = [r for r in records if query.lower() in str(r).lower()]
    return f"Found {len(results)} results: {results}"


db_tool = Tool(
    name="DatabaseSearch",
    func=search_database,
    description=(
        "Search the internal database for user records. "
        "Input is a SQL-like query string."
    ),
)

if __name__ == "__main__":
    print(db_tool.invoke(DEMO_QUERY_NAME))
    print(db_tool.invoke(DEMO_QUERY_EMAIL))
