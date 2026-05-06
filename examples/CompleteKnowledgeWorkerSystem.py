"""
Complete Multi-Agent Knowledge Worker System

A three-agent pipeline that researches, writes, and evaluates documentation
in an iterative loop:
  1. Researcher (AgentExecutor) — gathers information from web, docs, examples
  2. Writer (AgentExecutor)     — creates structured markdown documentation
  3. Evaluator (LCEL chain)     — scores on 5 dimensions and approves/rejects

The Writer and Evaluator loop up to max_iterations times until the Evaluator
returns approved=true (overall_score >= 8.0) or the limit is reached.
Output is saved to knowledge_worker_output.md.
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

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE          = 0      # deterministic LLM for researcher and evaluator
LLM_TEMPERATURE_CREATIVE = 0.7    # creative temperature for writer LLM
API_VERSION              = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
RESEARCHER_MAX_ITER      = 10     # max steps for researcher agent
WRITER_MAX_ITER          = 10     # max steps for writer agent
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
        "langchain agents": "LangChain agents use ReAct pattern for reasoning. Support tools like search, calculator, retrieval.",
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
        "api": "API documentation: Use create_react_agent() with llm, tools, and prompt parameters",
        "tools": "Tool creation: Use @tool decorator or Tool class with name, func, description",
        "callbacks": "Callbacks: Implement BaseCallbackHandler for monitoring and logging",
    }
    for key in docs:
        if key in query.lower():
            return f"Documentation: {docs[key]}"
    return "Documentation not found"


@tool
def search_examples(query: str) -> str:
    """Search code examples and tutorials."""
    examples = {
        "agent": "Example: agent = create_react_agent(llm, tools, prompt)",
        "multi": "Example: Orchestrate agents with sequential or hierarchical patterns",
        "monitoring": "Example: Use callbacks for logging and metrics collection",
    }
    for key in examples:
        if key in query.lower():
            return f"Code Example: {examples[key]}"
    return "No examples found"


# Create researcher tools and agent
researcher_tools = [search_web, search_documentation, search_examples]
researcher_prompt = hub.pull("hwchase17/react")
researcher_agent = create_react_agent(llm, researcher_tools, researcher_prompt)
researcher_executor = AgentExecutor(
    agent=researcher_agent,
    tools=researcher_tools,
    verbose=True,
    max_iterations=RESEARCHER_MAX_ITER,
    handle_parsing_errors=True,
)


# ============================================================================
# WRITER AGENT
# ============================================================================

@tool
def create_document_structure(topic: str) -> str:
    """Create a structured outline for documentation."""
    structure = f"""Document Structure for: {topic}

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
    return structure


@tool
def format_markdown(content: str) -> str:
    """Format content as professional markdown."""
    formatted = f"""# Documentation

{content}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    return formatted


@tool
def add_code_examples(section: str) -> str:
    """Generate code examples for a section."""
    examples = {
        "agent": """```python
from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools, prompt)
```""",
        "tools": """```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    \"\"\"Tool description.\"\"\"
    return process(input)
```""",
        "multi": """```python
result1 = agent1.invoke({"input": task})
result2 = agent2.invoke({"input": result1['output']})
```""",
    }
    for key in examples:
        if key in section.lower():
            return examples[key]
    return "# Code example placeholder"


# Create writer tools and agent
writer_tools = [create_document_structure, format_markdown, add_code_examples]
writer_prompt = hub.pull("hwchase17/react")
writer_agent = create_react_agent(llm_creative, writer_tools, writer_prompt)
writer_executor = AgentExecutor(
    agent=writer_agent,
    tools=writer_tools,
    verbose=True,
    max_iterations=WRITER_MAX_ITER,
    handle_parsing_errors=True,
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
        # JsonOutputParser handles markdown fences automatically
        self.chain = self.eval_prompt | self.llm | JsonOutputParser()

    def evaluate(self, content: str) -> Dict:
        """Evaluate content and return scores."""
        try:
            evaluation = self.chain.invoke({"content": content})
            return evaluation
        except Exception as e:
            # Fallback if parsing fails
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

        research_result = self.researcher.invoke({"input": research_task})
        research_findings = research_result["output"]

        print(f"\n📚 Research Complete. Findings length: {len(research_findings)} chars")

        self.history.append({
            "phase": "research",
            "output": research_findings,
        })

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

Revise the documentation addressing all feedback points. Improve sections with low scores."""

            writing_result = self.writer.invoke({"input": writing_task})
            current_document = writing_result["output"]

            print(f"\n📝 Document Complete. Length: {len(current_document)} chars")

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

            print(f"\n🎯 Evaluation Results:")
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
                print(f"\n🔄 Document needs improvement. Starting iteration {iteration + 1}...")

        # Final result
        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)

        if approved:
            print("✅ Documentation APPROVED and ready for publication!")
        else:
            print(f"⚠️  Documentation reached max iterations ({self.max_iterations}). Review needed.")

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
        researcher=researcher_executor,
        writer=writer_executor,
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

    print(f"\n📄 Document saved to: {OUTPUT_FILE}")

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
