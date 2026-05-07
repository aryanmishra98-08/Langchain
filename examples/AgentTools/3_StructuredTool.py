# =============================================================================
# Section 2.2 — Creating Tools: Method 3 — StructuredTool for complex inputs
# Topic:  StructuredTool.from_function() with a Pydantic v2 input schema for
#         tools that require multiple or typed parameters.
# =============================================================================
# Use StructuredTool when your tool accepts more than one parameter, or when
# you need field-level validation and per-field descriptions that the agent
# can use to construct correct inputs.
#
# The Pydantic BaseModel serves as the args_schema: the agent sees each field's
# description and type when deciding how to fill in the tool call arguments.
# =============================================================================

from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import StructuredTool

from pydantic import BaseModel, Field

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
DEMO_QUERY       = "langchain agents"     # search query for demo
DEMO_MAX_RESULTS = 3                      # max results for first demo call
DEMO_QUERY_2     = "multi-agent systems"  # second demo query (uses default max_results)
# ──────────────────────────────────────────────────────────────────────────────


class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results to return")


def advanced_search(query: str, max_results: int) -> str:
    """Simulate an advanced search with a configurable result count."""
    all_results = [f"Result {i} for '{query}'" for i in range(1, 11)]
    limited = all_results[:max_results]
    return str(limited)


advanced_search_tool = StructuredTool.from_function(
    func=advanced_search,
    name="AdvancedSearch",
    description="Search with advanced options",
    args_schema=SearchInput,
)

if __name__ == "__main__":
    # Invoke with both parameters
    result = advanced_search_tool.invoke({"query": DEMO_QUERY, "max_results": DEMO_MAX_RESULTS})
    print(result)

    # Invoke with default max_results
    result2 = advanced_search_tool.invoke({"query": DEMO_QUERY_2})
    print(result2)
