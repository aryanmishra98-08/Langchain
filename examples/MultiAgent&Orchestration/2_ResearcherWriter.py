# =============================================================================
# Section 5.2 — Multi-Agent Pattern 2: Researcher + Writer
# Topic:  A Researcher agent gathers information from web, news, and stats
#         tools; a Writer agent then creates a structured article using outline
#         and formatting tools. Demonstrates sequential (pipeline) pattern with
#         domain-separated agents.
# =============================================================================
# Communication pattern: Agent1 → Result → Agent2
#
# Researcher tools: search_papers, search_news, search_statistics
# Writer tools:     create_outline, format_article
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_agent

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
DEMO_TOPIC      = "The impact of AI on productivity"  # research and article topic
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


# ── Researcher tools ──────────────────────────────────────────────────────────

@tool
def search_papers(query: str) -> str:
    """Search academic papers and research."""
    papers = {
        "ai": "Study shows AI improves productivity by 40%",
        "climate": "New research indicates 2°C warming by 2050",
        "health": "Meta-analysis confirms benefits of exercise",
    }
    for key in papers:
        if key in query.lower():
            return papers[key]
    return "No relevant papers found"


@tool
def search_news(query: str) -> str:
    """Search recent news articles."""
    return f"Recent news on {query}: Major developments announced yesterday"


@tool
def search_statistics(topic: str) -> str:
    """Find statistical data on a topic."""
    return f"Statistics for {topic}: 65% growth year-over-year, 1.2M users"


# ── Writer tools ──────────────────────────────────────────────────────────────

@tool
def create_outline(topic: str) -> str:
    """Create article outline."""
    return f"""Outline for {topic}:
    I. Introduction
    II. Background
    III. Key Findings
    IV. Implications
    V. Conclusion"""


@tool
def format_article(content: str) -> str:
    """Format content as a professional article."""
    return f"""# Article

{content}

---
*Published: 2026*"""


# ── Researcher Agent ──────────────────────────────────────────────────────────

researcher_agent = create_agent(
    model=llm,
    tools=[search_papers, search_news, search_statistics],
    system_prompt=(
        "You are a research specialist. Gather comprehensive information from "
        "available tools and compile a detailed research summary."
    ),
)


# ── Writer Agent ──────────────────────────────────────────────────────────────

writer_agent = create_agent(
    model=llm,
    tools=[create_outline, format_article],
    system_prompt=(
        "You are a professional writer. Create well-structured, engaging articles "
        "based on provided research. Always create an outline first, then write the article."
    ),
)


# ── Orchestration ─────────────────────────────────────────────────────────────

def researcher_writer_system(topic: str) -> str:
    """Multi-agent research and writing system."""

    print("=" * 80)
    print("PHASE 1: RESEARCH")
    print("=" * 80)

    # Phase 1: Research
    research_task = f"""Research the topic: {topic}

Find:
1. Academic papers or studies
2. Recent news
3. Relevant statistics

Compile all findings into a comprehensive research summary."""

    research_result = researcher_agent.invoke({"messages": [{"role": "user", "content": research_task}]})
    research_findings = research_result["messages"][-1].content

    print("\n" + "=" * 80)
    print("RESEARCH FINDINGS:")
    print("=" * 80)
    print(research_findings)

    print("\n" + "=" * 80)
    print("PHASE 2: WRITING")
    print("=" * 80)

    # Phase 2: Write article
    writing_task = f"""Write a professional article on: {topic}

Use this research:
{research_findings}

Steps:
1. Create an outline
2. Write the article based on the outline and research
3. Format the article professionally

Create a complete, well-structured article."""

    writing_result = writer_agent.invoke({"messages": [{"role": "user", "content": writing_task}]})

    return writing_result["messages"][-1].content


if __name__ == "__main__":
    article = researcher_writer_system(DEMO_TOPIC)
    print("\n" + "=" * 80)
    print("FINAL ARTICLE:")
    print("=" * 80)
    print(article)
