"""
Complete Multi-Agent Knowledge Worker System

A three-agent pipeline that researches, writes, and evaluates documentation
in an iterative quality loop:

  1. Researcher (create_agent) — gathers information using web, docs, and
     examples search tools; produces a research findings summary
  2. Writer (create_agent)     — turns findings into structured markdown
     documentation using structure, formatting, and code example tools
  3. Evaluator (LCEL chain)    — scores the document on 5 dimensions
     (completeness, clarity, technical accuracy, examples, structure)
     and either approves (overall_score >= 8.0) or requests revisions

The Writer–Evaluator loop repeats up to max_iterations times. The final
document is saved to knowledge_worker_output.md.

Input:  {"messages": [{"role": "user", "content": "..."}]}
Output: result["messages"][-1].content
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI

from langchain.agents import create_agent

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE          = 0      # deterministic LLM for researcher and evaluator
LLM_TEMPERATURE_CREATIVE = 0.7    # creative temperature for writer LLM
API_VERSION              = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
SYSTEM_MAX_ITERATIONS    = 3      # max write-evaluate loop iterations
APPROVAL_THRESHOLD       = 8.0    # evaluator overall_score required for approval
OUTPUT_FILE              = "knowledge_worker_output.md"  # path to save final document
DEMO_TOPIC               = "LangChain Agents: Creating and Using Tools"  # topic to document
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)
llm_creative = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE_CREATIVE,
)


# ============================================================================
# RESEARCHER AGENT
# ============================================================================

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    results = {
        "langchain agents": "LangChain agents use tool-calling APIs for reasoning. Support tools like search, calculator, retrieval.",
        "multi-agent": "Multi-agent systems coordinate multiple AI agents with different roles and capabilities.",
        "production": "Production agents require monitoring, error handling, and guardrails for reliability.",
    }
    for key in results:
        if key in query.lower():
            return f"Web Search Result: {results[key]}"
    return "No relevant results found"


@tool
def search_documentation(query: str) -> str:
    """Search technical documentation."""
    docs = {
        "api": "API documentation: Use create_agent() with model, tools, system_prompt parameters",
        "tools": "Tool creation: Use @tool decorator or StructuredTool with name, func, description",
        "middleware": "Middleware: Implement BaseMiddleware for monitoring, guardrails, and loop control",
    }
    for key in docs:
        if key in query.lower():
            return f"Documentation: {docs[key]}"
    return "Documentation not found"


@tool
def search_examples(query: str) -> str:
    """Search code examples and tutorials."""
    examples = {
        "agent": "Example: agent = create_agent(model=llm, tools=tools, system_prompt='...')",
        "multi": "Example: Orchestrate agents with sequential or hierarchical patterns",
        "monitoring": "Example: Use MonitoringMiddleware for metrics collection",
    }
    for key in examples:
        if key in query.lower():
            return f"Code Example: {examples[key]}"
    return "No examples found"


researcher_agent = create_agent(
    model=llm,
    tools=[search_web, search_documentation, search_examples],
    system_prompt=(
        "You are a research specialist. Gather comprehensive information from "
        "all available tools and compile detailed research findings."
    ),
)


# ============================================================================
# WRITER AGENT
# ============================================================================

@tool
def create_document_structure(topic: str) -> str:
    """Create a structured outline for documentation."""
    return f"""Document Structure for: {topic}

1. Introduction
   - Overview
   - Key Concepts

2. Core Content
   - Main Topics
   - Technical Details

3. Examples
   - Code Samples
   - Use Cases

4. Best Practices
   - Recommendations
   - Common Pitfalls

5. Conclusion
   - Summary
   - Next Steps"""


@tool
def format_markdown(content: str) -> str:
    """Format content as professional markdown."""
    return f"""# Documentation

{content}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""


@tool
def add_code_examples(section: str) -> str:
    """Generate code examples for a section."""
    examples = {
        "agent": """```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```""",
        "tools": """```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    \"\"\"Tool description.\"\"\"
    return process(input)
```""",
        "multi": """```python
result1 = agent1.invoke({"messages": [{"role": "user", "content": task}]})
output1 = result1["messages"][-1].content
result2 = agent2.invoke({"messages": [{"role": "user", "content": output1}]})
```""",
    }
    for key in examples:
        if key in section.lower():
            return examples[key]
    return "# Code example placeholder"


writer_agent = create_agent(
    model=llm_creative,
    tools=[create_document_structure, format_markdown, add_code_examples],
    system_prompt=(
        "You are a technical writer. Create clear, well-structured documentation "
        "using the available tools. Always structure before writing."
    ),
)


# ============================================================================
# EVALUATOR (LCEL Chain — no agent needed for structured evaluation)
# ============================================================================

class EvaluatorAgent:
    """Agent that evaluates content quality."""

    def __init__(self, llm):
        self.llm = llm
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality evaluator. Assess documentation on:
1. Completeness (0-10): Does it cover all necessary topics?
2. Clarity (0-10): Is it easy to understand?
3. Technical Accuracy (0-10): Is the technical content correct?
4. Examples (0-10): Are there sufficient code examples?
5. Structure (0-10): Is it well-organized?

Provide scores and specific feedback for improvement.

Return your evaluation in this JSON format (no markdown, just raw JSON):
{{
    "completeness": <score>,
    "clarity": <score>,
    "technical_accuracy": <score>,
    "examples": <score>,
    "structure": <score>,
    "overall_score": <average>,
    "feedback": "<specific suggestions>",
    "approved": <true/false>
}}

Approve (true) if overall_score >= 8.0, otherwise reject (false)."""),
            ("human", "Evaluate this documentation:\n\n{content}"),
        ])

        # LCEL chain with built-in JSON parser
        self.chain = self.eval_prompt | self.llm | JsonOutputParser()

    def evaluate(self, content: str) -> Dict:
        try:
            return self.chain.invoke({"content": content})
        except Exception as e:
            return {
                "overall_score": 5.0,
                "completeness": 5,
                "clarity": 5,
                "technical_accuracy": 5,
                "examples": 5,
                "structure": 5,
                "feedback": f"Could not parse evaluation: {str(e)}",
                "approved": False,
            }


evaluator = EvaluatorAgent(llm)


# ============================================================================
# ORCHESTRATION
# ============================================================================

class KnowledgeWorkerSystem:
    """Complete multi-agent knowledge worker system."""

    def __init__(self, researcher, writer, evaluator, max_iterations=3):
        self.researcher = researcher
        self.writer = writer
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.history = []

    def process(self, topic: str) -> Dict:
        """Execute the full knowledge worker pipeline."""

        print("=" * 80)
        print(f"KNOWLEDGE WORKER SYSTEM: {topic}")
        print("=" * 80)

        # Phase 1: Research
        print("\n" + "=" * 80)
        print("PHASE 1: RESEARCH")
        print("=" * 80)

        research_task = f"""Research the topic: {topic}

Find information from:
1. Web sources (current information)
2. Technical documentation (API details)
3. Code examples (implementation patterns)

Compile comprehensive research findings covering all aspects of the topic."""

        research_result = self.researcher.invoke(
            {"messages": [{"role": "user", "content": research_task}]}
        )
        research_findings = research_result["messages"][-1].content

        print(f"\nResearch Complete. Findings length: {len(research_findings)} chars")

        self.history.append({"phase": "research", "output": research_findings})

        # Phase 2: Write (with iteration)
        iteration = 0
        approved = False
        current_document = None
        evaluation = None

        while iteration < self.max_iterations and not approved:
            print("\n" + "=" * 80)
            print(f"PHASE 2: WRITING (Iteration {iteration + 1})")
            print("=" * 80)

            if iteration == 0:
                writing_task = f"""Create comprehensive documentation on: {topic}

Use this research:
{research_findings}

Requirements:
1. Create a clear structure using the create_document_structure tool
2. Write detailed content for each section
3. Add relevant code examples using add_code_examples tool
4. Format professionally using format_markdown tool

Create complete, well-structured documentation."""
            else:
                writing_task = f"""Improve the documentation on: {topic}

Current version:
{current_document}

Evaluator feedback:
{evaluation['feedback']}

Research to reference:
{research_findings}

Revise the documentation addressing all feedback points."""

            writing_result = self.writer.invoke(
                {"messages": [{"role": "user", "content": writing_task}]}
            )
            current_document = writing_result["messages"][-1].content

            print(f"\nDocument Complete. Length: {len(current_document)} chars")

            self.history.append({
                "phase": "writing",
                "iteration": iteration + 1,
                "output": current_document,
            })

            # Phase 3: Evaluation
            print("\n" + "=" * 80)
            print(f"PHASE 3: EVALUATION (Iteration {iteration + 1})")
            print("=" * 80)

            evaluation = self.evaluator.evaluate(current_document)

            print(f"\nEvaluation Results:")
            print(f"   Overall Score: {evaluation['overall_score']}/10")
            print(f"   Completeness: {evaluation['completeness']}/10")
            print(f"   Clarity: {evaluation['clarity']}/10")
            print(f"   Technical Accuracy: {evaluation['technical_accuracy']}/10")
            print(f"   Examples: {evaluation['examples']}/10")
            print(f"   Structure: {evaluation['structure']}/10")
            print(f"   Approved: {evaluation['approved']}")
            print(f"\n   Feedback: {evaluation['feedback']}")

            self.history.append({
                "phase": "evaluation",
                "iteration": iteration + 1,
                "evaluation": evaluation,
            })

            approved = evaluation["approved"]
            iteration += 1

            if not approved and iteration < self.max_iterations:
                print(f"\nDocument needs improvement. Starting iteration {iteration + 1}...")

        # Final result
        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)

        if approved:
            print("Documentation APPROVED and ready for publication!")
        else:
            print(f"Documentation reached max iterations ({self.max_iterations}). Review needed.")

        return {
            "topic": topic,
            "final_document": current_document,
            "final_evaluation": evaluation,
            "approved": approved,
            "iterations": iteration,
            "history": self.history,
        }


# ============================================================================
# RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    system = KnowledgeWorkerSystem(
        researcher=researcher_agent,
        writer=writer_agent,
        evaluator=evaluator,
        max_iterations=SYSTEM_MAX_ITERATIONS,
    )

    result = system.process(DEMO_TOPIC)

    # Display final document
    print("\n" + "=" * 80)
    print("FINAL APPROVED DOCUMENT")
    print("=" * 80)
    print(result["final_document"])

    # Save to file
    with open(OUTPUT_FILE, "w") as f:
        f.write(result["final_document"])

    print(f"\nDocument saved to: {OUTPUT_FILE}")

    # Display statistics
    print("\n" + "=" * 80)
    print("SYSTEM STATISTICS")
    print("=" * 80)
    print(f"Total Iterations: {result['iterations']}")
    print(f"Final Score: {result['final_evaluation']['overall_score']}/10")
    print(f"Approved: {result['approved']}")
    if result["history"]:
        print(f"Research Findings: {len(result['history'][0]['output'])} chars")
    print(f"Final Document: {len(result['final_document'])} chars")
