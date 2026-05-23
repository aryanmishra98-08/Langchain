# LangChain Agents & Multi-Agent Systems: A Progressive Learning Guide

This repository teaches LangChain's agent system through 27 runnable examples organized into 6 progressive tracks using Azure OpenAI as the backend. Each example builds on the last, taking you from a simple LCEL chain to a production-ready three-agent knowledge worker pipeline.

**Target Audience:** Backend developers with Python and basic LangChain/RAG experience moving into agent development
**Duration:** ~3 hours hands-on (27 examples × ~6 min each)
**Prerequisites:** Python 3.10+, LangChain fundamentals, familiarity with RAG concepts

---

## Philosophy

LangChain's agent system is powerful, but the jump from "hello world" chains to production-ready multi-agent systems is steep. This repository distills the framework into **27 focused, runnable examples** organized into 6 progressive learning tracks. The guiding principles are:

- **Learn by doing.** Every concept has a dedicated, self-contained file you can run immediately.
- **Progressive complexity.** Each track introduces exactly one new layer — tools, single agents, observability, orchestration, production hardening — so the mental model builds cleanly.
- **Production awareness.** Guardrails, deduplication middleware, loop detection, and structured logging are first-class topics, not afterthoughts.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Structure](#repository-structure)
3. [What is LangChain?](#what-is-langchain)
4. [Track 1: Foundations](#track-1-foundations-examples-12)
5. [Track 2: Agent Tools](#track-2-agent-tools-examples-36)
6. [Track 3: Single Agent Development](#track-3-single-agent-development-examples-79)
7. [Track 4: Agent Reasoning & Observability](#track-4-agent-reasoning--observability-examples-1013)
8. [Track 5: Multi-Agent Orchestration](#track-5-multi-agent-orchestration-examples-1417)
9. [Track 6: Production Patterns](#track-6-production-patterns-examples-1823)
10. [Core Concepts at a Glance](#core-concepts-at-a-glance)
11. [Navigating the Examples](#navigating-the-examples)
12. [Multi-Agent Pattern Comparison](#multi-agent-pattern-comparison)
13. [LangChain 1.x Migration Notes](#langchain-1x-migration-notes)
14. [Additional Resources](#additional-resources)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Azure OpenAI account (with a deployed chat model and an embeddings model)

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd Langchain
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `keys/.env.example` to `keys/.env` and fill in your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002
```

All examples load this file automatically via:

```python
from dotenv import load_dotenv
load_dotenv("keys/.env")
```

### 3. Run your first example

```bash
python "examples/Foundations/1_LCELChain.py"
```

If you see a short summarized text printed to the terminal, you're ready to go.

---

## Repository Structure

```
Langchain/
├── README.md                                         ← You are here
├── requirements.txt                                  ← Python dependencies
├── LICENSE                                           ← Apache 2.0
├── keys/
│   └── .env                                          ← Your API keys (not committed)
└── examples/
    ├── Foundations/
    │   ├── 1_LCELChain.py                            ← Deterministic LCEL chain with pipe operator
    │   └── 2_ReactAgent.py                           ← Basic agent with dynamic tool selection
    ├── AgentTools/
    │   ├── 1_ToolDecorator.py                        ← @tool decorator, descriptions, error handling
    │   ├── 2_ToolClass.py                            ← Tool class for wrapping existing functions
    │   ├── 3_StructuredTool.py                       ← StructuredTool with Pydantic validation
    │   └── 4_BuiltinTools.py                         ← Calculator, web search, retrieval
    ├── SingleAgent/
    │   ├── 1_FirstAgent.py                           ← Complete first agent with two tools
    │   ├── 2_AgentTypes.py                           ← Standard vs specialist via system_prompt
    │   └── 3_AgentConfig.py                          ← Full create_agent parameter reference
    ├── AgentReasoning&Observability/
    │   ├── 1_VerboseMode.py                          ← Reading the Thought→Action→Observation trace
    │   ├── 2_IntermediateSteps.py                    ← Extracting (tool, input, output) step tuples
    │   ├── 3_CustomCallbacks.py                      ← BaseCallbackHandler for lifecycle events
    │   └── 4_ProductionLogging.py                    ← Structured file logging to timestamped files
    ├── MultiAgent&Orchestration/
    │   ├── 1_PlannerWorker.py                        ← Sequential: Planner chain + Worker agent
    │   ├── 2_ResearcherWriter.py                     ← Domain-separated sequential agents
    │   ├── 3_CriticBuilder.py                        ← Iterative refinement feedback loop
    │   └── 4_ManagerWorker.py                        ← Hierarchical dispatch with named workers
    ├── ProductionPatterns/
    │   ├── 1_OverInvocation.py                       ← DeduplicationMiddleware
    │   ├── 2_RunawayLoops.py                         ← LoopGuardMiddleware + async timeout
    │   ├── 3_ParsingErrors.py                        ← Why tool-calling eliminates parse errors
    │   ├── 4_ToolErrors.py                           ← Safe error handling and cross-platform timeouts
    │   ├── 5_Monitoring.py                           ← AgentMonitor + MonitoringMiddleware
    │   └── 6_Guardrails.py                           ← GuardrailAgent with input/output filtering
    └── CompleteKnowledgeWorkerSystem.py              ← Capstone: full 3-agent research-write-evaluate pipeline
```

---

## What is LangChain?

**LangChain** is a framework for building applications powered by language models. It simplifies LLM-driven workflows by providing composable abstractions for chains, agents, tools, and memory instead of writing prompt management, tool dispatch, and error handling from scratch.

### Why use LangChain?

Without it, you would need to:
- Manually format prompts, manage chat history, and parse LLM output across every call
- Write your own tool dispatch logic to route the LLM's output to the right function
- Implement retry logic, error handling, and timeouts around every external API call
- Build observability tooling to track what an agent actually did across a multi-step run

LangChain abstracts all of this into a unified, composable interface.

### Core building blocks

| Concept | Description |
|---------|-------------|
| **Runnable** | Base interface for anything invokable — chains, models, tools, parsers |
| **LCEL** | LangChain Expression Language; composes Runnables with the `\|` pipe operator |
| **Tool** | A named, described callable the agent can invoke at runtime |
| **Agent** | An LLM loop that selects and calls tools until it produces a final answer |
| **Middleware** | Hooks that intercept the agent loop for observability and flow control |
| **Callback** | Event listeners fired on lifecycle events (action, finish, tool start/end) |

---

## Track 1: Foundations (Examples 1–2)

### Theory

At its core, LangChain gives you two ways to connect an LLM to external functions: **chains** and **agents**. A chain is a pre-defined, deterministic sequence — every invocation follows the same path. An agent is a reasoning loop that decides at runtime which tools to call and in what order. The key abstraction is the **ReAct pattern** (Reasoning + Acting): the LLM thinks about what to do next, calls a tool, observes the result, and repeats until it can produce a final answer.

| Component | Role | Analogy |
|---|---|---|
| `PromptTemplate` | Formats input into a prompt | A form with fill-in blanks |
| `LLM` | Reasons and selects actions | The brain |
| `Tool` | Callable the agent dispatches to | A function the brain can call |
| `StrOutputParser` | Extracts plain text from model output | A decoder |

Chains are deterministic, fast, and cost-predictable — use them when the workflow is fixed. Agents are dynamic and judgment-driven — use them when the right sequence of steps depends on the input.

---

### Example 1 — LCEL Chain

**File:** `examples/Foundations/1_LCELChain.py`

Demonstrates a deterministic chain built with the pipe operator: input always flows `PromptTemplate → LLM → StrOutputParser` with no branching.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    PromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"text": "Long text..."})
print(result)
```

- `|` is the LCEL pipe operator; it connects any two `Runnable` objects into a sequential chain
- `StrOutputParser()` strips the `AIMessage` wrapper and returns a plain string

**Run it:**
```bash
python "examples/Foundations/1_LCELChain.py"
```

---

### Example 2 — ReAct Agent

**File:** `examples/Foundations/2_ReactAgent.py`

Introduces the agent loop: the model reads tool descriptions, selects the right tool, observes the result, and continues until it has a final answer.

```python
from langchain.agents import create_agent
from langchain_core.tools import Tool

tools = [search_tool, calculator_tool]

agent = create_agent(model=llm, tools=tools,
                     system_prompt="Use tools to answer questions accurately.")

result = agent.invoke({"messages": [
    {"role": "user", "content": "What's the population of NYC times 2?"}
]})
print(result["messages"][-1].content)
```

- Tool descriptions drive selection — the agent embeds them in its prompt and reasons about which fits the query
- `result["messages"][-1].content` is always the final text answer

**Run it:**
```bash
python "examples/Foundations/2_ReactAgent.py"
```

---

## Track 2: Agent Tools (Examples 3–6)

### Theory

A tool in LangChain is a `Runnable` with three parts: a **name** (how the agent refers to it), a **description** (what the agent reads to decide when to use it), and a **callable** that does the actual work and returns a string. There are three ways to create tools, each suited to a different situation.

**`@tool` decorator — for new functions:**

```python
@tool
def safe_calculator(expression: str) -> str:
    """Evaluate a math expression safely. Input: a valid expression like '2 + 2' or 'sqrt(16)'."""
    try:
        return f"Result: {numexpr.evaluate(expression).item()}"
    except Exception as e:
        return f"Error: {e}"
```

**`Tool` class — for wrapping existing functions:**

```python
db_tool = Tool(
    name="DatabaseSearch",
    func=search_database,
    description="Search internal user records. Input is a search string."
)
```

---

### Example 3 — Tool Decorator

**File:** `examples/AgentTools/1_ToolDecorator.py`

Shows the quality gap between vague and specific descriptions and demonstrates safe math evaluation using `numexpr` instead of `eval()`.

```python
@tool
def get_user_profile(user_id: str) -> str:
    """Retrieve user profile including name, email, and registration date.
    Use when you need detailed information about a specific user.
    Input must be a valid UUID user ID."""
    user = users_db.get(user_id)
    if not user:
        return f"Error: User '{user_id}' not found."
    return f"Name: {user['name']}, Email: {user['email']}"
```

- `numexpr.evaluate()` parses through a restricted grammar — unlike `eval()`, it has no code execution escape vectors
- The docstring body becomes the tool's `description` field verbatim

**Run it:**
```bash
python "examples/AgentTools/1_ToolDecorator.py"
```

---

### Example 4 — Tool Class

**File:** `examples/AgentTools/2_ToolClass.py`

Demonstrates wrapping an existing `search_database` function into a tool without modifying the original function.

```python
from langchain_core.tools import Tool

db_tool = Tool(
    name="DatabaseSearch",
    func=search_database,
    description="Search the internal database for user records. Input is a query string."
)

print(db_tool.invoke("name:Alice"))
```

- `Tool` is the right choice when you already have a function you cannot decorate
- `func=` accepts any callable that takes a single string and returns a string

**Run it:**
```bash
python "examples/AgentTools/2_ToolClass.py"
```

---

### Example 5 — Structured Tool

**File:** `examples/AgentTools/3_StructuredTool.py`

Uses a Pydantic `BaseModel` to give a tool multiple typed parameters with field-level descriptions — the model receives a proper JSON schema instead of a flat string.

```python
class SearchInput(BaseModel):
    query: str = Field(description="The search query to run")
    max_results: int = Field(default=5, description="Maximum number of results to return")

advanced_search = StructuredTool.from_function(
    func=run_search,
    name="AdvancedSearch",
    description="Search with control over result count.",
    args_schema=SearchInput
)

print(advanced_search.invoke({"query": "LangChain agents", "max_results": 3}))
```

- `args_schema=SearchInput` generates a JSON schema the model uses to populate arguments
- `Field(default=5)` makes a parameter optional without breaking the schema

**Run it:**
```bash
python "examples/AgentTools/3_StructuredTool.py"
```

---

### Example 6 — Built-in Tools

**File:** `examples/AgentTools/4_BuiltinTools.py`

Demonstrates three production-ready tools: a sandboxed calculator, a DuckDuckGo web search, and a Chroma-backed retrieval tool.

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool

search = DuckDuckGoSearchRun()
retrieval_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(),
    name="document_search",
    description="Search internal documentation. Use for company-specific questions."
)
```

- `DuckDuckGoSearchRun` requires no API key and is a drop-in for internet queries
- `create_retriever_tool` wraps any LangChain retriever — swap Chroma for any other vector store

**Run it:**
```bash
python "examples/AgentTools/4_BuiltinTools.py"
```

---

## Track 3: Single Agent Development (Examples 7–9)

### Theory

`create_agent` is the standard factory for building agents in LangChain 1.x. It wires together an LLM, a list of tools, and a system prompt into a runnable agent that uses the model's **native tool-calling API** — the LLM outputs structured JSON specifying which tool to call and with what arguments, eliminating the fragile text parsing of older agent patterns. The agent's response is always a dict with a `"messages"` key containing the full conversation, including every tool call and its result.

**Step-by-step data flow:**

```
User message
        ↓  LLM reasons over message + tool schemas
AIMessage(tool_calls=[{name, args}])
        ↓  Framework dispatches to matching tool
ToolMessage(content=tool_output)
        ↓  LLM observes result, reasons again
AIMessage(content=final_answer)   ← result["messages"][-1]
```

---

### Example 7 — First Agent

**File:** `examples/SingleAgent/1_FirstAgent.py`

Builds a complete working agent with a `calculator` tool and a `company_knowledge` retrieval tool, then runs three queries that exercise calculation-only, retrieval-only, and both tools in one run.

```python
agent = create_agent(
    model=llm,
    tools=[calculator, company_knowledge],
    system_prompt="You are a helpful assistant. Use available tools to answer questions accurately."
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is 15 * 67?"}]})
print(result["messages"][-1].content)
```

- `company_knowledge` is a `create_retriever_tool` over a 4-document Chroma store seeded inline
- The agent decides at runtime which tool (or combination) fits each query

**Run it:**
```bash
python "examples/SingleAgent/1_FirstAgent.py"
```

---

### Example 8 — Agent Types

**File:** `examples/SingleAgent/2_AgentTypes.py`

Shows how changing `system_prompt` alone produces measurably different agent behavior — no code changes, no new tools.

```python
standard_agent = create_agent(model=llm, tools=tools,
    system_prompt="You are a helpful assistant.")

specialist_agent = create_agent(model=llm, tools=tools,
    system_prompt=(
        "You are a math specialist. When given a calculation, always use "
        "the calculator tool and show the exact expression you evaluated."
    ))
```

- Both agents share the same LLM and tools — `system_prompt` is the only variable
- The specialist's instruction changes both tool selection behavior and output format

**Run it:**
```bash
python "examples/SingleAgent/2_AgentTypes.py"
```

---

### Example 9 — Agent Config

**File:** `examples/SingleAgent/3_AgentConfig.py`

Documents every `create_agent` parameter with inline explanations and demonstrates how to read tool call metadata from the reasoning trace.

```python
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
    name="config_demo_agent",
    middleware=[MonitoringMiddleware(monitor)],
)

result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
for msg in result["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"Called: {tc['name']}({tc['args']})")
```

- `name=` appears in LangSmith traces and is required for multi-agent routing
- `middleware=` accepts a list of `BaseMiddleware` instances for observability and flow control

**Run it:**
```bash
python "examples/SingleAgent/3_AgentConfig.py"
```

---

## Track 4: Agent Reasoning & Observability (Examples 10–13)

### Theory

Every agent invocation returns `result["messages"]` — a list of `HumanMessage`, `AIMessage`, and `ToolMessage` objects that constitute the full reasoning trace. Reading this list is the primary way to understand what the agent decided and why. For production systems you need two additional layers: **callbacks** (per-event listeners for dev-time inspection and third-party integrations) and **structured logging** (durable records of every action and tool output for post-hoc auditing).

```
result["messages"] structure
        ↓
HumanMessage        — original user query
AIMessage           — agent reasoning + tool_calls=[{name, args, id}]
ToolMessage         — tool output, matched to AIMessage via tool_call_id
...                 — repeated for each tool invoked
AIMessage           — final answer (no tool_calls, only content)
```

---

### Example 10 — Verbose Mode

**File:** `examples/AgentReasoning&Observability/1_VerboseMode.py`

Implements `print_reasoning_trace()` to format `result["messages"]` into a human-readable Thought → Action → Observation → Answer transcript.

```python
def print_reasoning_trace(messages):
    for msg in messages:
        label = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[Action] {tc['name']}({tc['args']})")
        elif msg.content:
            print(f"[{label}] {msg.content}")
```

- `type(msg).__name__` returns `"HumanMessage"`, `"AIMessage"`, or `"ToolMessage"` without importing each class
- `ToolMessage.content` holds the raw string returned by the tool

**Run it:**
```bash
python "examples/AgentReasoning&Observability/1_VerboseMode.py"
```

---

### Example 11 — Intermediate Steps

**File:** `examples/AgentReasoning&Observability/2_IntermediateSteps.py`

Correlates AIMessage tool calls with their ToolMessage results using `tool_call_id` to produce structured `{tool, input, output}` dicts for auditing and metrics.

```python
def extract_steps(messages):
    tool_outputs = {m.tool_call_id: m.content
                    for m in messages if hasattr(m, "tool_call_id")}
    steps = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", []):
            steps.append({
                "tool": tc["name"],
                "input": tc["args"],
                "output": tool_outputs.get(tc["id"], "")
            })
    return steps
```

- Matching on `tool_call_id` / `tc["id"]` is necessary because a single AIMessage can contain multiple parallel tool calls
- The returned list of dicts is suitable for logging, dashboards, or cost attribution

**Run it:**
```bash
python "examples/AgentReasoning&Observability/2_IntermediateSteps.py"
```

---

### Example 12 — Custom Callbacks

**File:** `examples/AgentReasoning&Observability/3_CustomCallbacks.py`

Implements `CustomAgentCallbackHandler` by subclassing `BaseCallbackHandler` to intercept four lifecycle events during a live run.

```python
class CustomAgentCallbackHandler(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        print(f"[Action] {action.tool}({action.tool_input})")

    def on_tool_end(self, output, **kwargs):
        print(f"[Observation] {output}")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config={"callbacks": [CustomAgentCallbackHandler()]}
)
```

- Callbacks fire **during** execution; `result["messages"]` is only available **after**
- Use callbacks for dev-time tracing and third-party integrations; use middleware for flow control

**Run it:**
```bash
python "examples/AgentReasoning&Observability/3_CustomCallbacks.py"
```

---

### Example 13 — Production Logging

**File:** `examples/AgentReasoning&Observability/4_ProductionLogging.py`

A `ProductionAgentCallback` that writes structured log entries to a timestamped file, capturing every action, tool output, and error with ISO timestamps.

```python
logging.basicConfig(
    filename=f'agent_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

- One log file per agent session keeps runs isolated for post-hoc comparison
- For zero-config full tracing, set `LANGCHAIN_TRACING_V2=true` to send traces to LangSmith instead

**Run it:**
```bash
python "examples/AgentReasoning&Observability/4_ProductionLogging.py"
```

---

## Track 5: Multi-Agent Orchestration (Examples 14–17)

### Theory

Multi-agent systems decompose complex tasks by assigning specialized agents to distinct sub-problems, then composing their outputs. The four patterns in this track cover the most common topologies: **sequential pipeline**, **domain-separated**, **iterative feedback loop**, and **hierarchical dispatch**. Agents communicate by passing the output of one invocation as the input message of the next — the simplest handoff is `result["messages"][-1].content` passed as the next agent's user message.

**Sequential handoff — step-by-step data flow:**

```
User task
        ↓  Agent 1 executes with its tools
Agent 1 output (plain string)
        ↓  passed as user message to Agent 2
Agent 2 output (plain string)
        ↓  passed as user message to Agent 3
Final result
```

---

### Example 14 — Planner + Worker

**File:** `examples/MultiAgent&Orchestration/1_PlannerWorker.py`

A two-step pipeline where a planning LCEL chain decomposes a task into a numbered list, then a worker agent executes each step using `fetch_data`, `process_data`, and `generate_report` tools.

```python
planner = ChatPromptTemplate.from_messages([
    ("system", "You are a planning expert. Break the task into numbered steps."),
    ("human", "{task}")
]) | llm | StrOutputParser()

plan = planner.invoke({"task": task})

worker = create_agent(model=llm, tools=[fetch_data, process_data, generate_report],
                      system_prompt="Execute the plan step by step.")
result = worker.invoke({"messages": [{"role": "user", "content": plan}]})
```

- The Planner is a pure LCEL chain — no tools needed for text-only planning
- Passing `plan` as the worker's user message is the simplest form of inter-agent handoff

**Run it:**
```bash
python "examples/MultiAgent&Orchestration/1_PlannerWorker.py"
```

---

### Example 15 — Researcher + Writer

**File:** `examples/MultiAgent&Orchestration/2_ResearcherWriter.py`

Two agents with completely different tool sets execute sequentially — the Researcher gathers raw findings, the Writer receives those findings as context and produces a formatted article.

```python
researcher = create_agent(model=llm,
    tools=[search_papers, search_news, search_statistics],
    system_prompt="Gather comprehensive research findings.")

writer = create_agent(model=llm,
    tools=[create_outline, format_article],
    system_prompt="Write a clear, well-structured article from the research provided.")

research = researcher.invoke({"messages": [{"role": "user", "content": topic}]})
article = writer.invoke({"messages": [
    {"role": "user", "content": f"Write an article using: {research['messages'][-1].content}"}
]})
```

- Separating tool sets enforces domain boundaries — the Writer cannot do research, the Researcher cannot format
- This pattern scales to any number of sequential specialists

**Run it:**
```bash
python "examples/MultiAgent&Orchestration/2_ResearcherWriter.py"
```

---

### Example 16 — Critic + Builder

**File:** `examples/MultiAgent&Orchestration/3_CriticBuilder.py`

An iterative refinement loop where a Builder chain generates code and a Critic chain either approves it (`APPROVED: <reason>`) or returns revision notes (`NEEDS_WORK: <issues>`). The loop exits on approval or after `max_iterations`.

```python
for i in range(max_iterations):
    code = builder.invoke({"requirements": requirements, "feedback": feedback})
    review = critic.invoke({"code": code})

    if review.startswith("APPROVED"):
        return {"status": "approved", "code": code, "iterations": i + 1}
    feedback = review

return {"status": "max_iterations_reached", "code": code}
```

- The `APPROVED:` / `NEEDS_WORK:` protocol is a simple, parseable signal — structured output schemas are an alternative for stricter parsing
- Both components are LCEL chains; agents are not needed when there is no tool use

**Run it:**
```bash
python "examples/MultiAgent&Orchestration/3_CriticBuilder.py"
```

---

### Example 17 — Manager + Workers

**File:** `examples/MultiAgent&Orchestration/4_ManagerWorker.py`

A `ManagerWorkerSystem` where a Manager LCEL chain decomposes a task into `ASSIGN: worker_name | task_description` directives and routes each subtask to a named Worker agent.

```python
class ManagerWorkerSystem:
    def run(self, task: str) -> dict:
        plan = self.manager.invoke({"task": task})
        assignments = self._parse_assignments(plan)
        results = {}
        for worker_name, subtask in assignments:
            worker = self.workers[worker_name]
            results[worker_name] = worker.invoke(
                {"messages": [{"role": "user", "content": subtask}]}
            )["messages"][-1].content
        return results
```

- `_parse_assignments()` parses the `ASSIGN:` lines — a lightweight protocol that avoids structured output overhead
- Naming workers in the Manager's system prompt is how routing is specified

**Run it:**
```bash
python "examples/MultiAgent&Orchestration/4_ManagerWorker.py"
```

---

## Track 6: Production Patterns (Examples 18–23 + Capstone)

### Theory

Running agents in production introduces failure modes that don't appear in demos: the same tool called with the same input five times, a loop that burns tokens for minutes, a tool that raises instead of returning a string, or a query that extracts PII from a response. This track addresses each systematically.

```
# Without guardrails
User: "explain exploit CVE-2024-0001"
Agent: [calls search tool repeatedly] → expensive, no answer   ← runaway loop
```

```
# With guardrails
User: "explain exploit CVE-2024-0001"
GuardrailAgent: prohibited term detected → "Query not allowed"  ← rejected before LLM call
```

#### How the modern middleware pattern works

The approach has three parts:

1. **`BaseMiddleware` subclass** — implement `before_model`, `after_model`, `after_tool`, or `on_error`
2. **Stateful tracking** — store seen tool calls, loop counts, or metrics in `__init__`
3. **Signal via exception or state mutation** — raise to abort the loop, or mutate state to redirect it

```
User message
        ↓
before_model(state, config)        ← DeduplicationMiddleware checks here
        ↓
LLM call
        ↓
after_model(state, config)
        ↓
Tool execution
        ↓
after_tool(state, config)          ← MonitoringMiddleware records here
        ↓
on_error(error, state, config)     ← fires if any step raises
```

#### The golden rule for tools

Every tool must return a string under all conditions — including on error. A tool that raises will crash the agent loop.

```python
@tool
def safe_calculator(expression: str) -> str:
    """Evaluate a math expression. Input: a valid expression string."""
    try:
        return f"Result: {numexpr.evaluate(expression).item()}"
    except ZeroDivisionError:
        return "Error: Division by zero. Please revise the expression."
    except Exception as e:
        return f"Error: {e}"
```

#### Middleware hooks reference

| Hook | When it fires |
|------|---------------|
| `before_model(state, config)` | Before each LLM call |
| `after_model(state, config)` | After each LLM call |
| `after_tool(state, config)` | After each tool execution |
| `on_error(error, state, config)` | On any exception in the loop |

---

### Example 18 — Over-invocation

**File:** `examples/ProductionPatterns/1_OverInvocation.py`

Demonstrates `DeduplicationMiddleware` that tracks `(tool_name, input)` pairs and aborts on the second identical call.

```python
class DeduplicationMiddleware(BaseMiddleware):
    def __init__(self):
        self.seen = set()

    def before_model(self, state, config):
        for msg in state["messages"]:
            for tc in getattr(msg, "tool_calls", []):
                key = (tc["name"], str(tc["args"]))
                if key in self.seen:
                    raise ValueError(f"Duplicate tool call: {tc['name']}")
                self.seen.add(key)
        return state, config
```

- Adding `"IMPORTANT: Only call this tool once per unique query."` to the description is the first, cheapest fix
- Middleware is the reliable backstop when prompt-level instructions are insufficient

**Run it:**
```bash
python "examples/ProductionPatterns/1_OverInvocation.py"
```

---

### Example 19 — Runaway Loops

**File:** `examples/ProductionPatterns/2_RunawayLoops.py`

Combines `LoopGuardMiddleware` (detects repeated tool observations), an explicit system-prompt rule, and an `asyncio.wait_for` timeout wrapper.

```python
async def invoke_with_timeout(agent, input_data, timeout=30):
    return await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None, lambda: agent.invoke(input_data)
        ),
        timeout=timeout
    )
```

- `LoopGuardMiddleware` raises when it sees the same `ToolMessage` content repeated — the earliest possible abort
- The async timeout is a hard wall-clock limit that catches any loop the middleware misses

**Run it:**
```bash
python "examples/ProductionPatterns/2_RunawayLoops.py"
```

---

### Example 20 — Parsing Errors

**File:** `examples/ProductionPatterns/3_ParsingErrors.py`

Explains why `OutputParserException` cannot occur with `create_agent`: the model outputs a JSON object specifying the tool call — there is no freeform text to parse.

```python
# Text-parsed agents (pre-1.x) had to parse output like:
# "Action: Calculator\nAction Input: 2 + 2"  ← fragile, format-sensitive

# create_agent receives structured JSON:
# {"name": "calculator", "args": {"expression": "2 + 2"}}  ← no parsing needed
agent = create_agent(model=llm, tools=[calculator],
                     system_prompt="Use the calculator for all math.")
```

- This is a structural guarantee from the tool-calling API, not a configuration setting
- For open-source LLMs without native tool-calling, LangGraph's `create_react_agent` with `handle_parsing_errors=True` is the fallback

**Run it:**
```bash
python "examples/ProductionPatterns/3_ParsingErrors.py"
```

---

### Example 21 — Tool Errors

**File:** `examples/ProductionPatterns/4_ToolErrors.py`

Demonstrates three patterns: per-exception error messages, pre-execution validation, and a cross-platform timeout using `concurrent.futures`.

```python
def run_with_timeout(func, args, timeout=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return f"Error: Operation timed out after {timeout} seconds."
```

- `concurrent.futures` works on Windows, Linux, and macOS — `signal.SIGALRM` is Unix-only
- Pre-validation (checking the expression before calling `numexpr`) produces better error messages than catching after

**Run it:**
```bash
python "examples/ProductionPatterns/4_ToolErrors.py"
```

---

### Example 22 — Monitoring

**File:** `examples/ProductionPatterns/5_Monitoring.py`

`AgentMonitor` accumulates metrics across runs; `MonitoringMiddleware` hooks into `after_tool` and `on_error` to populate it automatically.

```python
monitor = AgentMonitor()
agent = create_agent(model=llm, tools=tools,
                     middleware=[MonitoringMiddleware(monitor)])

for query in demo_queries:
    agent.invoke({"messages": [{"role": "user", "content": query}]})

print(monitor.get_report())
# {"total_calls": 3, "success_rate": 1.0, "avg_execution_time": 2.4,
#  "tool_usage": {"calculator": 2, "search": 1}, "recent_errors": []}
```

- `tool_usage` lets you identify which tools are hot paths worth optimizing
- `recent_errors` caps at 5 entries to avoid unbounded memory growth

**Run it:**
```bash
python "examples/ProductionPatterns/5_Monitoring.py"
```

---

### Example 23 — Guardrails

**File:** `examples/ProductionPatterns/6_Guardrails.py`

`GuardrailAgent` wraps any agent with pre-execution input checks (query length, prohibited terms) and post-execution output checks (PII regex patterns).

```python
rules = {
    "max_query_length": 500,
    "prohibited_terms": ["hack", "exploit", "bypass"],
    "allow_pii": False
}

guarded = GuardrailAgent(base_agent, rules)
result = guarded.invoke("What is 2 + 2?")
# → {"allowed": True, "result": "4"}

result = guarded.invoke("How do I hack a system?")
# → {"allowed": False, "reason": "Query contains prohibited term: 'hack'"}
```

- Pre-execution checks short-circuit before the LLM is called, saving both cost and latency
- PII detection uses regex for speed; for higher recall, replace with a dedicated classifier

**Run it:**
```bash
python "examples/ProductionPatterns/6_Guardrails.py"
```

---

### Example 24 — Knowledge Worker System (Capstone)

**File:** `examples/CompleteKnowledgeWorkerSystem.py`

Brings every concept together into a production-quality `KnowledgeWorkerSystem` class. Features:

- Three-agent pipeline: Researcher (`create_agent`) → Writer (`create_agent`) → Evaluator (LCEL chain)
- Iterative quality loop: Evaluator scores on five dimensions; loop continues until score ≥ 8.0 or `max_iterations` reached
- Structured evaluation via `JsonOutputParser` returning per-dimension scores and actionable feedback
- Final document written to `knowledge_worker_output.md` with run statistics

```python
class KnowledgeWorkerSystem:
    def __init__(self, llm, max_iterations=3):
        self.researcher = create_agent(model=llm,
            tools=[search_web, search_documentation, search_examples],
            system_prompt="You are a thorough researcher.")
        self.writer = create_agent(model=llm,
            tools=[create_document_structure, format_markdown, add_code_examples],
            system_prompt="You are a technical writer. Produce clear, accurate documentation.")
        self.evaluator = eval_prompt | llm | JsonOutputParser()

    def run(self, topic: str) -> dict:
        research = self._research(topic)
        for i in range(self.max_iterations):
            document = self._write(topic, research, feedback)
            evaluation = self.evaluator.invoke({"document": document})
            if evaluation["overall_score"] >= 8.0:
                break
            feedback = evaluation["feedback"]
        return {"document": document, "score": evaluation["overall_score"]}
```

**Run it:**
```bash
python "examples/CompleteKnowledgeWorkerSystem.py"
```

The system prints phase-by-phase progress, evaluation scores, and a final status line. The finished document is saved to `knowledge_worker_output.md` in the working directory.

---

## Core Concepts at a Glance

| Concept | What It Does | Where to Start |
|---------|-------------|----------------|
| **LCEL pipe operator** | Composes Runnables into deterministic chains | Example 1 |
| **ReAct loop** | LLM reasons, calls a tool, observes, repeats | Example 2 |
| **`@tool` decorator** | Turns a Python function into a named, described tool | Example 3 |
| **`StructuredTool`** | Gives a tool a Pydantic schema for multi-parameter inputs | Example 5 |
| **`create_agent`** | Factory for building native-tool-calling agents | Example 7 |
| **`system_prompt`** | Controls agent behavior, persona, and tool usage rules | Example 8 |
| **Reasoning trace** | `result["messages"]` — the full Thought→Action→Observation sequence | Example 10 |
| **`BaseCallbackHandler`** | Per-event listener for dev-time inspection and integrations | Example 12 |
| **`BaseMiddleware`** | State-aware hook for flow control and production observability | Examples 18–22 |
| **Multi-agent handoff** | Passing `result["messages"][-1].content` as the next agent's input | Examples 14–17 |
| **Iterative refinement** | Feedback loop between builder and critic until approval | Example 16 |
| **`GuardrailAgent`** | Wrapper enforcing input/output safety rules before and after execution | Example 23 |

---

## Navigating the Examples

**If you're brand new to LangChain agents:** Start at Example 1 and work through the tracks in order — each track assumes the previous one.

**If you already understand LCEL and basic tool creation:** Jump to Track 3 (Example 7) and work forward through observability and multi-agent patterns.

**If you need to debug a specific production issue:** Go directly to the relevant ProductionPatterns example — each file is self-contained and names the problem in its filename.

**If you want a production reference:** The capstone at `examples/CompleteKnowledgeWorkerSystem.py` demonstrates every pattern integrated into a single working system.

**Running any example:**
```bash
python "examples/<track_folder>/<filename>.py"
```

All examples load environment variables from `keys/.env` automatically.

---

## Multi-Agent Pattern Comparison

| Pattern | Topology | Components | Example | Use When |
|---------|----------|------------|---------|----------|
| **Planner + Worker** | Sequential | LCEL chain → Agent | Example 14 | Task needs decomposition before execution |
| **Researcher + Writer** | Sequential | Agent → Agent | Example 15 | Domain specialization with different tool sets |
| **Critic + Builder** | Iterative loop | Chain ↔ Chain | Example 16 | Output quality requires iterative refinement |
| **Manager + Workers** | Hierarchical | Chain → N Agents | Example 17 | Task parallelizes across named sub-domains |

---

## LangChain 1.x Migration Notes

### Old approach (pre-1.x `initialize_agent`)

```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
result = agent.run("What is 2 + 2?")
```

### New approach (LangChain 1.x `create_agent`)

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant."
)
result = agent.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})
print(result["messages"][-1].content)
```

`create_agent` uses the model's native tool-calling API instead of text-parsed ReAct, which eliminates `OutputParserException` and produces more reliable tool selection. All examples in this repository use the 1.x pattern.

**Note on LangGraph:** This repository uses `create_agent` throughout. LangGraph (the modern state-machine orchestration layer) enables persistent state, parallel branches, and human-in-the-loop workflows and is covered in a separate session.

---

## Additional Resources

- [LangChain 1.x Concepts Reference](https://python.langchain.com/docs/concepts/)
- [LangChain Agents How-To Guide](https://python.langchain.com/docs/how_to/#agents)
- [LangChain Tools How-To Guide](https://python.langchain.com/docs/how_to/#tools)
- [LangSmith Tracing Documentation](https://docs.smith.langchain.com/)
- [LangChain Community Discord](https://discord.gg/langchain)

---

<p align="center">
  Built for learning. Designed for production readiness.<br>
  Licensed under Apache 2.0.
</p>
