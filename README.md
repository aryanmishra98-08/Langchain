# LangChain Agents & Multi-Agent Systems: Advanced Training Guide

**Target Audience:** Developers with LangChain basics + RAG experience  
**Level:** Advanced  
**Prerequisites:** Python 3.10+, LangChain fundamentals, RAG concepts  
**LangChain Version:** 0.3.x (Classic AgentExecutor pattern)

> **Note on LangGraph:** This guide uses the classic `AgentExecutor` pattern. LangGraph (the modern orchestration framework that supersedes manual multi-agent code) is covered in a later session.

---

## Table of Contents

1. [Conceptual Foundations](#1-conceptual-foundations)
2. [Tools and Tool Creation](#2-tools-and-tool-creation)
3. [Single Agent Development](#3-single-agent-development)
4. [Agent Reasoning & Transparency](#4-agent-reasoning--transparency)
5. [Multi-Agent Architectures](#5-multi-agent-architectures)
6. [Production Debugging & Pitfalls](#6-production-debugging--pitfalls)
7. [Advanced Capstone Project](#7-advanced-capstone-project)
8. [References & Resources](#8-references--resources)

---

## Setup & Installation

All examples in this guide assume the following package versions:

```bash
pip install \
  "langchain>=0.3,<0.4" \
  "langchain-core>=0.3" \
  "langchain-openai>=0.2" \
  "langchain-community>=0.3" \
  "langchain-chroma>=0.1.4" \
  "langchainhub>=0.1.20" \
  "numexpr>=2.10" \
  "pydantic>=2.7"

export OPENAI_API_KEY="your-key-here"
```

See [requirements.txt](requirements.txt) for the pinned dependency list.

> **Why separate packages?** Since LangChain 0.2 (mid-2024), the framework was split: `langchain-core` (stable interfaces), `langchain` (orchestration), and provider-specific packages (`langchain-openai`, `langchain-chroma`, etc.). Always import from the most specific package available.

---

## 1. Conceptual Foundations

→ Examples: [examples/01_foundations/](examples/01_foundations/)

### 1.1 Agent vs Chain: Understanding the Difference

**Chain:**
- Deterministic, pre-defined sequence of operations
- Fixed control flow: A → B → C
- Predictable execution path
- Lower cost, faster execution
- Better for well-defined workflows

A chain using LCEL (LangChain Expression Language) looks like this — see [examples/01_foundations/01_chain_example.py](examples/01_foundations/01_chain_example.py):

```python
chain = (
    PromptTemplate.from_template("Summarize: {text}")
    | llm
    | StrOutputParser()
)
result = chain.invoke({"text": "Long text..."})  # Always follows same path
```

> **Migration note:** The legacy `LLMChain` class is deprecated and scheduled for removal. LCEL (the `|` pipe operator) is now the standard way to compose chains. It provides better streaming, batching, async support, and observability.

**Agent:**
- Dynamic reasoning and decision-making
- Non-deterministic control flow
- Can choose tools based on context
- Higher cost, variable execution time
- Better for open-ended tasks requiring judgment

See [examples/01_foundations/02_agent_example.py](examples/01_foundations/02_agent_example.py) for a basic ReAct agent that dynamically selects between search and calculator tools.

**Key Decision Matrix:**

| Use Case | Chain | Agent |
|----------|-------|-------|
| Fixed workflow (RAG query → retrieve → answer) | ✅ | ❌ |
| Dynamic tool selection needed | ❌ | ✅ |
| Budget-sensitive application | ✅ | ❌ |
| Requires reasoning about which action to take | ❌ | ✅ |
| Production-critical path (high reliability) | ✅ | ⚠️ |

### 1.2 How Agents Work: The ReAct Pattern

LangChain agents typically use the **ReAct** (Reasoning + Acting) pattern:

```
Thought: I need to find the population of NYC
Action: search[population of NYC]
Observation: NYC has 8.3 million people
Thought: Now I need to multiply by 2
Action: calculator[8.3 * 2]
Observation: 16.6
Thought: I have the final answer
Final Answer: 16.6 million
```

**ReAct Loop:**
1. **Think:** Reason about what to do next
2. **Act:** Use a tool or provide final answer
3. **Observe:** Process tool output
4. Repeat until satisfied

### 1.3 Tool Selection Logic

Agents use tool descriptions to decide which tool to invoke:

```python
search_tool = Tool(
    name="Search",
    func=search_function,
    description="Useful for finding current information about events, people, or facts. Input should be a search query."
)

calculator_tool = Tool(
    name="Calculator",
    func=calculator_function,
    description="Useful for mathematical calculations. Input should be a mathematical expression like '2 + 2' or '15 * 67'."
)
```

**Agent decision process:**
1. Parse user question
2. Match intent to tool descriptions
3. Select most relevant tool
4. Execute and observe result
5. Continue or terminate

### 1.4 When to Use Agents (and When Not To)

**✅ Use Agents When:**
- Task requires dynamic decision-making
- Multiple tools might be needed in unknown order
- User queries are open-ended
- Examples: Research assistants, data analysis, complex Q&A

**❌ Avoid Agents When:**
- Workflow is well-defined and fixed
- Latency is critical
- Costs must be minimized
- Reliability is paramount (agents can hallucinate actions)
- Examples: Simple classification, fixed RAG pipelines, batch processing

---

## 2. Tools and Tool Creation

→ Examples: [examples/02_tools/](examples/02_tools/)

### 2.1 Understanding Tools

A tool in LangChain is a callable interface that:
1. Has a name (for agent reference)
2. Has a description (for agent selection)
3. Has a defined input schema
4. Returns a string output (for agent observation)

### 2.2 Creating Tools from Python Functions

**Method 1: Using the `@tool` decorator** — see [examples/02_tools/01_tool_decorator.py](examples/02_tools/01_tool_decorator.py)

The docstring becomes the tool description; type hints define the input schema:

```python
@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return str(a * b)
```

**Method 2: Using the `Tool` class directly** — see [examples/02_tools/02_tool_class.py](examples/02_tools/02_tool_class.py)

```python
db_tool = Tool(
    name="DatabaseSearch",
    func=search_database,
    description="Search the internal database for user records. Input is a SQL-like query string."
)
```

**Method 3: Using `StructuredTool` for complex inputs (Pydantic v2)** — see [examples/02_tools/03_structured_tool.py](examples/02_tools/03_structured_tool.py)

```python
class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results to return")

advanced_search_tool = StructuredTool.from_function(
    func=advanced_search,
    name="AdvancedSearch",
    description="Search with advanced options",
    args_schema=SearchInput
)
```

> **Migration note:** Imports moved from `langchain.tools` to `langchain_core.tools`. The `langchain.tools` path still works via re-export but emits deprecation warnings. `StructuredTool.from_function()` is the recommended factory; direct constructor usage is being phased out.

### 2.3 Best Practices for Tool Creation

**1. Clear, Specific Descriptions**

```python
# ❌ Bad
@tool
def get_data(id: str) -> str:
    """Gets data."""
    return fetch(id)

# ✅ Good
@tool
def get_user_profile(user_id: str) -> str:
    """Retrieves user profile information including name, email, and registration date.

    Use this when you need detailed information about a specific user.
    Input should be a valid user ID (format: UUID).
    """
    return fetch_user(user_id)
```

**2. Error Handling**

```python
@tool
def safe_calculator(expression: str) -> str:
    """Safely evaluate mathematical expressions. ..."""
    try:
        result = numexpr.evaluate(expression).item()
        return f"Result: {result}"
    except Exception as e:
        return f"Error: Could not evaluate expression. {str(e)}"
```

> **Why numexpr?** `eval()` — even with restricted `__builtins__` — has known escape vectors and is unsafe for production. `numexpr` parses expressions through a restricted grammar that only supports mathematical operations, making it the standard choice for production calculator tools.

**3. Consistent Return Format**

Tools should always return a human-readable string, including on errors:

```python
@tool
def weather_lookup(city: str) -> str:
    """Get current weather for a city."""
    try:
        ...
        return f"Temperature: {weather.temp}°F, Conditions: {weather.conditions}"
    except CityNotFound:
        return f"Error: City '{city}' not found. Please check spelling."
```

### 2.4 Built-in Tool Examples

See [examples/02_tools/04_builtin_tools.py](examples/02_tools/04_builtin_tools.py) for ready-to-use implementations of:

- **Calculator Tool** — `numexpr`-based sandboxed math evaluation
- **Web Search Tool** — `DuckDuckGoSearchRun` (requires internet)
- **Retrieval Tool** — `create_retriever_tool` from a Chroma vector store (requires `OPENAI_API_KEY`)

> **Migration notes:**
> - `Chroma` moved from `langchain_community.vectorstores` to its own package: `pip install langchain-chroma` and `from langchain_chroma import Chroma`.
> - `create_retriever_tool` moved from `langchain.tools.retriever` to `langchain_core.tools`.
> - Use `text-embedding-3-small` (cheaper, better) instead of the legacy default `text-embedding-ada-002`.

> **Note on PythonREPL:** The legacy `PythonREPL` from `langchain_community.utilities` (or its newer home in `langchain_experimental`) executes arbitrary Python code and is a significant security risk. Avoid exposing it to agents handling untrusted input. Use focused tools like `numexpr`-based calculators instead.

---

## 3. Single Agent Development

→ Examples: [examples/03_single_agent/](examples/03_single_agent/)

### 3.1 Building Your First Agent

See [examples/03_single_agent/01_first_agent.py](examples/03_single_agent/01_first_agent.py) for a complete working agent with:
- A `calculator` tool (numexpr, production-safe)
- A `company_knowledge` retrieval tool (Chroma vector store)
- Three test queries covering: calculation only, retrieval only, and both tools combined

### 3.2 Agent Types in LangChain

See [examples/03_single_agent/02_agent_types.py](examples/03_single_agent/02_agent_types.py) for side-by-side implementations of all three types.

**1. ReAct Agent** — works with any LLM via prompt-engineered text parsing:

```python
from langchain.agents import create_react_agent
from langchain import hub

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
```

**2. Tool Calling Agent** — recommended for modern LLMs, uses native tool-calling API:

```python
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
```

**3. Structured Chat Agent** — for multi-input tools without native tool-calling support:

```python
from langchain.agents import create_structured_chat_agent

prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
```

> **Migration note:** `create_openai_functions_agent` is deprecated. Use `create_tool_calling_agent` instead — it's model-agnostic (works with OpenAI, Anthropic, Google, etc.) and uses each provider's native tool-calling API. This is more reliable than text-parsed ReAct and produces fewer parsing errors.

**Choosing an agent type:**

| Agent Type | Best For | Reliability |
|------------|----------|-------------|
| ReAct | Open-source / non-tool-calling LLMs | ⚠️ Medium (text parsing) |
| Tool Calling | Modern OpenAI / Anthropic / Google models | ✅ High (native API) |
| Structured Chat | Multi-input tools on legacy models | ⚠️ Medium |

### 3.3 Agent Configuration

See [examples/03_single_agent/03_agent_config.py](examples/03_single_agent/03_agent_config.py) for an annotated `AgentExecutor` with all key parameters.

**Key `AgentExecutor` parameters:**

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                    # Show agent reasoning
    handle_parsing_errors=True,      # Gracefully handle LLM output errors
    max_iterations=10,               # Prevent infinite loops
    max_execution_time=60,           # Timeout in seconds
    early_stopping_method="force",   # "force" returns "Stopped" message
    return_intermediate_steps=True   # Get full reasoning trace in output
)
```

> **Note on `early_stopping_method`:** Only `"force"` is reliably supported. The `"generate"` option requires the agent class to implement a special return method and is not supported by all agent types.

---

## 4. Agent Reasoning & Transparency

→ Examples: [examples/04_reasoning/](examples/04_reasoning/)

### 4.1 Understanding Agent Thought Process

See [examples/04_reasoning/01_verbose_mode.py](examples/04_reasoning/01_verbose_mode.py).

Setting `verbose=True` on the `AgentExecutor` prints the full ReAct trace to stdout:

```
> Entering new AgentExecutor chain...
I need to find the company's revenue first, then the number of employees, then divide.

Action: company_knowledge
Action Input: revenue

Observation: Annual revenue for 2023 was $5 million.
Thought: Now I need to find the number of employees.

Action: company_knowledge
Action Input: employees

Observation: We have 50 employees across 3 offices.
Thought: Now I can calculate revenue per employee.

Action: calculator
Action Input: 5000000 / 50

Observation: The result is: 100000.0
Thought: I now know the final answer.
Final Answer: The revenue per employee is $100,000.

> Finished chain.
```

### 4.2 Capturing Intermediate Steps

See [examples/04_reasoning/02_intermediate_steps.py](examples/04_reasoning/02_intermediate_steps.py).

Set `return_intermediate_steps=True` to access the reasoning chain programmatically. Each step is a `(AgentAction, observation)` tuple:

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
result = agent_executor.invoke({"input": "What is the revenue per employee?"})

for step in result['intermediate_steps']:
    action, observation = step
    print(f"Action: {action.tool}")
    print(f"Input: {action.tool_input}")
    print(f"Output: {observation}")
```

### 4.3 Custom Callbacks for Monitoring

See [examples/04_reasoning/03_custom_callbacks.py](examples/04_reasoning/03_custom_callbacks.py) for a `CustomAgentCallbackHandler` that captures every action and finish event.

Callbacks intercept the following lifecycle events:
- `on_agent_action` — called when the agent selects a tool
- `on_agent_finish` — called when the agent produces its final answer
- `on_tool_start` — called just before tool execution begins
- `on_tool_end` — called with the tool's string output

```python
result = agent_executor.invoke(
    {"input": "..."},
    config={"callbacks": [callback_handler]}
)
```

> **Migration note:** `from langchain.callbacks.base` → `from langchain_core.callbacks`. Callbacks are now passed via `config={"callbacks": [...]}` in `.invoke()` calls (the LCEL standard), though direct `callbacks=[...]` parameters still work for backward compatibility.

### 4.4 Logging for Production

See [examples/04_reasoning/04_production_logging.py](examples/04_reasoning/04_production_logging.py) for a `ProductionAgentCallback` that writes structured log entries to a timestamped file with `verbose=False` for clean console output.

```python
logging.basicConfig(
    filename=f'agent_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

> **Pro tip:** For production observability, consider LangSmith (`LANGCHAIN_TRACING_V2=true` environment variable). It captures full traces — including LLM calls, token usage, and latencies — without writing custom callbacks.

---

## 5. Multi-Agent Architectures

→ Examples: [examples/05_multi_agent/](examples/05_multi_agent/)

### 5.1 Pattern 1: Planner + Worker

See [examples/05_multi_agent/01_planner_worker.py](examples/05_multi_agent/01_planner_worker.py).

One agent (LCEL chain) plans; another (AgentExecutor) executes. The Planner returns a numbered list; the Worker receives the full plan as its task:

```
STEP 1: PLANNING  →  Planner chain produces numbered plan
STEP 2: EXECUTION →  Worker agent executes each step with tools
```

The Planner is a pure LCEL chain (no `AgentExecutor` needed for text-only planning). The Worker uses `fetch_data`, `process_data`, and `generate_report` tools.

### 5.2 Pattern 2: Researcher + Writer

See [examples/05_multi_agent/02_researcher_writer.py](examples/05_multi_agent/02_researcher_writer.py).

Domain-separated agents with different tool sets:
- **Researcher** tools: `search_papers`, `search_news`, `search_statistics`
- **Writer** tools: `create_outline`, `format_article`

```
PHASE 1: RESEARCH  →  Researcher gathers findings
PHASE 2: WRITING   →  Writer receives findings and produces article
```

### 5.3 Pattern 3: Critic + Builder

See [examples/05_multi_agent/03_critic_builder.py](examples/05_multi_agent/03_critic_builder.py).

An iterative feedback loop using two LCEL chains:
- **Builder** generates code from requirements (or revises based on feedback)
- **Critic** scores the code and either approves (`APPROVED: <reason>`) or requests changes (`NEEDS_WORK: <issues>`)

The loop continues until approved or `max_iterations` is reached:

```python
result = critic_builder_loop(requirements, max_iterations=3)
# result["status"] → "approved" or "max_iterations_reached"
```

### 5.4 Communication Patterns

**Pattern A: Sequential (Pipeline)**
```
Agent1 → Result → Agent2 → Result → Agent3
```

**Pattern B: Hierarchical (Manager-Worker)**
```
        Manager
       /   |   \
Worker1 Worker2 Worker3
```

**Pattern C: Peer-to-Peer (Debate)**
```
Agent1 ←→ Agent2
   ↓         ↓
     Mediator
```

See [examples/05_multi_agent/04_manager_worker.py](examples/05_multi_agent/04_manager_worker.py) for a `ManagerWorkerSystem` class that implements the hierarchical pattern. The Manager LCEL chain parses a high-level task into `ASSIGN: worker_name | task_description` directives and dispatches subtasks to named Worker AgentExecutors.

---

## 6. Production Debugging & Pitfalls

→ Examples: [examples/06_production/](examples/06_production/)

### 6.1 Common Issues and Solutions

#### Issue 1: Over-invocation of Tools

See [examples/06_production/01_over_invocation.py](examples/06_production/01_over_invocation.py).

**Problem:** Agent calls the same tool repeatedly with similar inputs.

**Symptoms:**
```
Action: search  →  Observation: $5M
Action: search  →  Observation: $5M   (same)
Action: search  →  Observation: $5M   (same)
```

**Solutions:**

1. **Limit iterations** — `AgentExecutor(max_iterations=5, ...)`
2. **`DeduplicationCallback`** — raises `ValueError` when the same `(tool, input)` pair is seen again
3. **Improve tool description** — add `"IMPORTANT: Only call this once per unique query."` to the docstring

#### Issue 2: Runaway Loops

See [examples/06_production/02_runaway_loops.py](examples/06_production/02_runaway_loops.py).

**Problem:** Agent enters an infinite loop without making progress when tools consistently return no results.

**Solutions:**

1. **Timeout** — `AgentExecutor(max_execution_time=60, ...)`
2. **`ProgressCallback`** — detects when the same observation appears `max_no_progress` times in a row and raises `ValueError`
3. **Enhanced system prompt** — explicit rules: "If a tool returns 'No results', DO NOT retry the same query"

#### Issue 3: Parsing Errors

See [examples/06_production/03_parsing_errors.py](examples/06_production/03_parsing_errors.py).

**Problem:** LLM returns malformed action/input text, raising `OutputParserException`.

**Solutions:**

1. **`handle_parsing_errors=True`** — automatic retry on parse failures
2. **Custom error handler** — a callable that returns a corrective instruction string
3. **`create_tool_calling_agent`** — eliminates text-parsing errors entirely (native JSON output)

> **Recommendation:** Most parsing errors disappear when you switch from `create_react_agent` (text-parsed) to `create_tool_calling_agent` (native structured output). Use ReAct only when your LLM doesn't support tool calling.

#### Issue 4: Tool Execution Errors

See [examples/06_production/04_tool_errors.py](examples/06_production/04_tool_errors.py).

**Problem:** Tools crash (e.g., `ZeroDivisionError`) or stall (external API timeouts).

**Solutions:**

1. **`safe_calculator`** — per-exception-type error messages returned as strings
2. **`validated_calculator`** — pre-validate the expression before passing to `numexpr`
3. **`run_with_timeout`** — cross-platform timeout via `concurrent.futures.ThreadPoolExecutor`

> **Migration note:** The `signal.SIGALRM` pattern from older guides is Unix-only. Use `concurrent.futures` for cross-platform compatibility.

### 6.2 Monitoring and Observability

See [examples/06_production/05_monitoring.py](examples/06_production/05_monitoring.py) for a complete `AgentMonitor` + `MonitoringCallback` implementation tracking:
- Total calls, success rate, average execution time
- Per-tool invocation counts
- Last 5 error records with timestamps

```python
monitor = AgentMonitor()
monitoring_callback = MonitoringCallback(monitor)
agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[monitoring_callback])

# After running queries...
print(monitor.get_report())
```

### 6.3 Guardrails and Constraints

See [examples/06_production/06_guardrails.py](examples/06_production/06_guardrails.py) for a `GuardrailAgent` wrapper with:
- **Pre-execution input checks:** max query length, prohibited term list
- **Post-execution output checks:** PII detection (email/phone regex patterns)

```python
rules = {
    'max_query_length': 500,
    'prohibited_terms': ['hack', 'exploit', 'bypass'],
    'allow_pii': False
}
guarded_agent = GuardrailAgent(agent_executor, rules)
```

---

## 7. Advanced Capstone Project

→ Example: [examples/07_capstone/capstone_knowledge_worker.py](examples/07_capstone/capstone_knowledge_worker.py)

### Multi-Agent Knowledge Worker System

**Scenario:** A complete three-agent pipeline that researches, writes, and evaluates documentation in an iterative quality loop.

**Agents:**
1. **Researcher** (`AgentExecutor`) — gathers information using `search_web`, `search_documentation`, `search_examples` tools
2. **Writer** (`AgentExecutor`) — creates structured markdown documentation using `create_document_structure`, `format_markdown`, `add_code_examples` tools
3. **Evaluator** (LCEL chain) — scores the document on 5 dimensions and approves or rejects it

**Pipeline:**
```
PHASE 1: RESEARCH       →  Researcher gathers findings
PHASE 2: WRITING (×N)  →  Writer drafts/revises document
PHASE 3: EVALUATION     →  Evaluator scores and approves/rejects
                            ↳ If rejected and iterations remain → back to PHASE 2
```

**Evaluator scoring dimensions (0–10 each):**

| Dimension | Description |
|---|---|
| Completeness | Does it cover all necessary topics? |
| Clarity | Is it easy to understand? |
| Technical Accuracy | Is the technical content correct? |
| Examples | Are there sufficient code examples? |
| Structure | Is it well-organized? |

Approval threshold: `overall_score >= 8.0`. Final document is saved to `knowledge_worker_output.md`.

### Running the Capstone

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python examples/07_capstone/capstone_knowledge_worker.py
```

### Expected Output Flow

```
================================================================================
KNOWLEDGE WORKER SYSTEM: LangChain Agents: Creating and Using Tools
================================================================================

================================================================================
PHASE 1: RESEARCH
================================================================================

> Entering new AgentExecutor chain...
...
📚 Research Complete. Findings length: 1247 chars

================================================================================
PHASE 2: WRITING (Iteration 1)
================================================================================
...
📝 Document Complete. Length: 2134 chars

================================================================================
PHASE 3: EVALUATION (Iteration 1)
================================================================================

🎯 Evaluation Results:
   Overall Score: 7.2/10
   ...
   Approved: False
   Feedback: Add more code examples in sections 2 and 3...

🔄 Document needs improvement. Starting iteration 2...
...
✅ Documentation APPROVED and ready for publication!
```

### Extension Ideas

1. **Add More Agents:** SEO Optimizer, Code Tester, Translator
2. **Enhanced Communication:** Shared memory between agents, parallel execution
3. **Production Features:** Database persistence, API endpoints, real-time progress updates, human-in-the-loop approval

---

## 8. References & Resources

### Official Documentation

- **LangChain Concepts**: https://python.langchain.com/docs/concepts/
- **Agents Guide**: https://python.langchain.com/docs/how_to/#agents
- **Tools Guide**: https://python.langchain.com/docs/how_to/#tools
- **AgentExecutor**: https://python.langchain.com/docs/how_to/agent_executor/
- **LCEL (LangChain Expression Language)**: https://python.langchain.com/docs/concepts/lcel/
- **LangSmith (observability)**: https://docs.smith.langchain.com/

### Modern Convention Quick Reference

| Legacy (avoid) | Modern (use) |
|---|---|
| `LLMChain` | LCEL: `prompt \| llm \| StrOutputParser()` |
| `chain.run(...)` | `chain.invoke({...})` |
| `from langchain.tools import tool` | `from langchain_core.tools import tool` |
| `from langchain.prompts import ...` | `from langchain_core.prompts import ...` |
| `from langchain.schema import ...` | `from langchain_core.messages import ...` |
| `from langchain.callbacks.base import ...` | `from langchain_core.callbacks import ...` |
| `from langchain_community.vectorstores import Chroma` | `from langchain_chroma import Chroma` |
| `create_openai_functions_agent` | `create_tool_calling_agent` |
| `eval()` for math | `numexpr.evaluate()` |
| `signal.SIGALRM` timeouts | `concurrent.futures` |
| `text-embedding-ada-002` | `text-embedding-3-small` |

### Best Practices

1. **Start Simple:** Begin with single agent, add complexity gradually
2. **Clear Tool Descriptions:** Crucial for agent decision-making
3. **Monitor Everything:** Use callbacks and logging extensively (or LangSmith)
4. **Set Limits:** Always use `max_iterations` and `max_execution_time`
5. **Handle Errors:** Wrap tools in try-except, use `handle_parsing_errors`
6. **Test Thoroughly:** Agents are non-deterministic, test edge cases
7. **Cost Management:** Monitor token usage, agents can be expensive
8. **Iterative Development:** Build → Test → Refine cycle
9. **Prefer Tool Calling Agents:** Use `create_tool_calling_agent` over ReAct when your model supports native tool calling
10. **Pin Versions:** LangChain APIs evolve quickly — pin minor versions in production

### Common Pitfalls to Avoid

❌ **Over-reliance on agents**: Not everything needs an agent  
❌ **Vague tool descriptions**: Leads to poor tool selection  
❌ **No iteration limits**: Can cause runaway costs  
❌ **Ignoring errors**: Tools should gracefully handle failures  
❌ **Missing monitoring**: Can't debug without visibility  
❌ **Complex multi-agent from start**: Build incrementally  
❌ **No human oversight**: Agents make mistakes, have review processes  
❌ **Using `eval()` in tools**: Production security risk — use `numexpr` for math  
❌ **Mixing legacy and modern imports**: Pick `langchain_core.*` consistently  

### Production Checklist

- [ ] Error handling in all tools
- [ ] Iteration and time limits set
- [ ] Comprehensive logging and monitoring
- [ ] Deduplication and loop prevention
- [ ] Cost tracking and alerts
- [ ] Security guardrails implemented
- [ ] Output validation
- [ ] Fallback mechanisms
- [ ] Human approval for critical actions
- [ ] A/B testing for agent changes
- [ ] Pinned package versions
- [ ] LangSmith tracing enabled (or equivalent)

### Advanced Topics for Further Study

- **LangGraph**: Modern orchestration framework for multi-agent systems (covered in next session)
- **Memory Systems**: Long-term memory for agents (`RunnableWithMessageHistory`)
- **Tool Chaining**: Complex tool compositions
- **Agent Fine-tuning**: Custom models for specific agent behaviors
- **Distributed Agents**: Scaling across multiple instances
- **Human-in-the-Loop**: Interactive agent workflows

---

## Project Structure

```
.
├── requirements.txt
├── examples/
│   ├── 01_foundations/
│   │   ├── 01_chain_example.py          # LCEL chain (Section 1.1)
│   │   └── 02_agent_example.py          # Basic ReAct agent (Sections 1.1, 1.3)
│   ├── 02_tools/
│   │   ├── 01_tool_decorator.py         # @tool decorator, best practices (Sections 2.2–2.3)
│   │   ├── 02_tool_class.py             # Tool class directly (Section 2.2)
│   │   ├── 03_structured_tool.py        # StructuredTool + Pydantic v2 (Section 2.2)
│   │   └── 04_builtin_tools.py          # Calculator, search, retrieval (Section 2.4)
│   ├── 03_single_agent/
│   │   ├── 01_first_agent.py            # Complete first agent (Section 3.1)
│   │   ├── 02_agent_types.py            # ReAct / Tool Calling / Structured Chat (Section 3.2)
│   │   └── 03_agent_config.py           # AgentExecutor parameters (Section 3.3)
│   ├── 04_reasoning/
│   │   ├── 01_verbose_mode.py           # verbose=True trace (Section 4.1)
│   │   ├── 02_intermediate_steps.py     # return_intermediate_steps (Section 4.2)
│   │   ├── 03_custom_callbacks.py       # CustomAgentCallbackHandler (Section 4.3)
│   │   └── 04_production_logging.py     # ProductionAgentCallback (Section 4.4)
│   ├── 05_multi_agent/
│   │   ├── 01_planner_worker.py         # Planner + Worker pattern (Section 5.1)
│   │   ├── 02_researcher_writer.py      # Researcher + Writer pattern (Section 5.2)
│   │   ├── 03_critic_builder.py         # Critic + Builder loop (Section 5.3)
│   │   └── 04_manager_worker.py         # Hierarchical Manager-Worker (Section 5.4)
│   ├── 06_production/
│   │   ├── 01_over_invocation.py        # Tool over-invocation fixes (Section 6.1)
│   │   ├── 02_runaway_loops.py          # Runaway loop fixes (Section 6.1)
│   │   ├── 03_parsing_errors.py         # Parsing error fixes (Section 6.1)
│   │   ├── 04_tool_errors.py            # Tool execution error fixes (Section 6.1)
│   │   ├── 05_monitoring.py             # AgentMonitor + MonitoringCallback (Section 6.2)
│   │   └── 06_guardrails.py             # GuardrailAgent (Section 6.3)
│   └── 07_capstone/
│       └── capstone_knowledge_worker.py # Full KnowledgeWorkerSystem (Section 7)
└── keys/
```

## Conclusion

This guide covers:
- ✅ Difference between Agents and Chains (with modern LCEL)
- ✅ Creating custom tools with proper descriptions and modern imports
- ✅ Building single agents with calculator and retrieval
- ✅ Understanding and logging agent reasoning
- ✅ Multi-agent patterns (Planner+Worker, Researcher+Writer, Critic+Builder, Hierarchical)
- ✅ Production debugging, monitoring, and guardrails
- ✅ Building a complete multi-agent knowledge worker system

**Next Steps:**
1. Implement the examples in your own projects
2. Experiment with different agent architectures
3. Build production-ready monitoring systems
4. **Next session: LangGraph** for advanced orchestration with state management

