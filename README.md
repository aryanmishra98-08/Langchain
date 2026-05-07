# LangChain Agents & Multi-Agent Systems

**Target Audience:** Developers with LangChain basics + RAG experience  
**Level:** Advanced  
**Prerequisites:** Python 3.10+, LangChain fundamentals, RAG concepts

> **Note on LangGraph:** This guide uses the `create_agent` pattern. LangGraph (the modern state-machine orchestration framework) is covered in a later session.

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

```bash
pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for the full pinned dependency list.

Copy `keys/.env.example` to `keys/.env` and fill in your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_CHAT_DEPLOYMENT=...
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=...
AZURE_OPENAI_API_VERSION=...
```

---

## 1. Conceptual Foundations

→ Examples: [examples/Foundations/](examples/Foundations/)

### 1.1 Agent vs Chain: Understanding the Difference

**Chain:**
- Deterministic, pre-defined sequence of operations
- Fixed control flow: A → B → C
- Predictable execution path
- Lower cost, faster execution
- Better for well-defined workflows

A chain using LCEL (LangChain Expression Language) looks like this — see [examples/Foundations/1_LCELChain.py](examples/Foundations/1_LCELChain.py):

```python
chain = (
    PromptTemplate.from_template("Summarize: {text}")
    | llm
    | StrOutputParser()
)
result = chain.invoke({"text": "Long text..."})  # Always follows the same path
```

> **Why LCEL?** The `|` pipe operator composes Runnables and is the standard way to build chains. It provides better streaming, batching, async support, and observability than class-based alternatives.

**Agent:**
- Dynamic reasoning and decision-making
- Non-deterministic control flow
- Chooses tools based on context
- Higher cost, variable execution time
- Better for open-ended tasks requiring judgment

See [examples/Foundations/2_ReactAgent.py](examples/Foundations/2_ReactAgent.py) for a basic agent that dynamically selects between search and calculator tools.

**Key Decision Matrix:**

| Use Case | Chain | Agent |
|----------|-------|-------|
| Fixed workflow (RAG query → retrieve → answer) | ✅ | ❌ |
| Dynamic tool selection needed | ❌ | ✅ |
| Budget-sensitive application | ✅ | ❌ |
| Requires reasoning about which action to take | ❌ | ✅ |
| Production-critical path (high reliability) | ✅ | ⚠️ |

### 1.2 How Agents Work: The ReAct Pattern

LangChain agents use the **ReAct** (Reasoning + Acting) pattern:

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

→ Examples: [examples/AgentTools/](examples/AgentTools/)

### 2.1 Understanding Tools

A tool in LangChain is a callable interface that:
1. Has a name (for agent reference)
2. Has a description (for agent selection)
3. Has a defined input schema
4. Returns a string output (for agent observation)

### 2.2 Creating Tools from Python Functions

**Method 1: Using the `@tool` decorator** — see [examples/AgentTools/1_ToolDecorator.py](examples/AgentTools/1_ToolDecorator.py)

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

**Method 2: Using the `Tool` class directly** — see [examples/AgentTools/2_ToolClass.py](examples/AgentTools/2_ToolClass.py)

```python
db_tool = Tool(
    name="DatabaseSearch",
    func=search_database,
    description="Search the internal database for user records. Input is a SQL-like query string."
)
```

**Method 3: Using `StructuredTool` for complex inputs** — see [examples/AgentTools/3_StructuredTool.py](examples/AgentTools/3_StructuredTool.py)

Use this when your tool accepts more than one parameter or needs field-level validation:

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

**Choosing a method:**

| Scenario | Method |
|----------|--------|
| Single string input, new function | `@tool` decorator |
| Wrapping an existing function | `Tool` class |
| Multiple typed parameters | `StructuredTool` with Pydantic |

### 2.3 Best Practices for Tool Creation

**1. Clear, Specific Descriptions**

```python
# ❌ Bad — agent cannot reliably select this
@tool
def get_data(id: str) -> str:
    """Gets data."""
    return fetch(id)

# ✅ Good — specific, actionable
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

> **Why numexpr?** `eval()` — even with restricted `__builtins__` — has known escape vectors. `numexpr` parses expressions through a restricted grammar that only allows mathematical operations, making it the safe choice for calculator tools.

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

See [examples/AgentTools/4_BuiltinTools.py](examples/AgentTools/4_BuiltinTools.py) for ready-to-use implementations of:

- **Calculator Tool** — `numexpr`-based sandboxed math evaluation
- **Web Search Tool** — `DuckDuckGoSearchRun` (requires internet)
- **Retrieval Tool** — `create_retriever_tool` from a Chroma vector store

> **Note on PythonREPL:** `PythonREPL` from `langchain_experimental` executes arbitrary Python code and is a significant security risk. Avoid it for agents handling untrusted input. Use focused tools like `numexpr`-based calculators instead.

---

## 3. Single Agent Development

→ Examples: [examples/SingleAgent/](examples/SingleAgent/)

### 3.1 Building Your First Agent

See [examples/SingleAgent/1_FirstAgent.py](examples/SingleAgent/1_FirstAgent.py) for a complete working agent with:
- A `calculator` tool (numexpr, production-safe)
- A `company_knowledge` retrieval tool (Chroma vector store)
- Three test queries covering: calculation only, retrieval only, and both tools combined

The standard pattern:

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the available tools to answer questions accurately.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
print(result["messages"][-1].content)
```

### 3.2 Agent Types

See [examples/SingleAgent/2_AgentTypes.py](examples/SingleAgent/2_AgentTypes.py).

`create_agent` uses the model's native tool-calling API (structured JSON), which is more reliable than text-parsed alternatives. Customize behavior via `system_prompt`:

```python
# Standard agent
standard_agent = create_agent(model=llm, tools=tools, system_prompt="You are a helpful assistant.")

# Specialist agent with custom behavior
specialist_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a math specialist. When given a calculation, always use the "
        "calculator tool and show the expression you evaluated."
    ),
)
```

> **For open-source LLMs without native tool-calling:** LangGraph's prebuilt `create_react_agent` is the fallback. This is covered in the LangGraph session.

### 3.3 Agent Configuration

See [examples/SingleAgent/3_AgentConfig.py](examples/SingleAgent/3_AgentConfig.py) for all `create_agent` parameters with inline explanations.

**Key parameters:**

| Parameter | Purpose |
|-----------|---------|
| `model` | LLM instance or model string |
| `tools` | List of tool functions / Tool objects |
| `system_prompt` | Agent instructions |
| `name` | Identifier for tracing in multi-agent systems |
| `middleware` | List of `BaseMiddleware` for observability and flow control |
| `state_schema` | Custom TypedDict extending AgentState |
| `response_format` | Constrain output to a specific schema |

**Using middleware and name:**

```python
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
    name="my_agent",                              # shows up in LangSmith traces
    middleware=[MonitoringMiddleware(monitor)],   # see ProductionPatterns/5_Monitoring.py
)
```

**Accessing the reasoning trace:**

```python
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

for msg in result["messages"]:
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            print(f"Tool call: {tc['name']}({tc['args']})")
    elif msg.content:
        print(f"[{type(msg).__name__}] {msg.content}")
```

---

## 4. Agent Reasoning & Transparency

→ Examples: [examples/AgentReasoning&Observability/](examples/AgentReasoning&Observability/)

### 4.1 Reading the Reasoning Trace

See [examples/AgentReasoning&Observability/1_VerboseMode.py](examples/AgentReasoning&Observability/1_VerboseMode.py).

`result["messages"]` contains the full conversation, including each tool call and its result. Iterating over this list gives you the Thought → Action → Observation → Answer sequence:

```
HumanMessage   — user's original question
AIMessage      — agent reasoning + tool_calls=[{name, args}]
ToolMessage    — tool result (observation)
...
AIMessage      — final answer (no tool_calls, just content)
```

### 4.2 Extracting Intermediate Steps

See [examples/AgentReasoning&Observability/2_IntermediateSteps.py](examples/AgentReasoning&Observability/2_IntermediateSteps.py).

Pair AIMessage tool calls with their corresponding ToolMessages using `tool_call_id`:

```python
for step in extract_steps(result["messages"]):
    print(f"Action: {step['tool']}")
    print(f"Input: {step['input']}")
    print(f"Output: {step['output']}")
```

This enables post-hoc auditing, custom metrics, and UIs that show reasoning progress.

### 4.3 Custom Callbacks for Monitoring

See [examples/AgentReasoning&Observability/3_CustomCallbacks.py](examples/AgentReasoning&Observability/3_CustomCallbacks.py) for a `CustomAgentCallbackHandler`.

Callbacks intercept the following lifecycle events:
- `on_agent_action` — called when the agent selects a tool
- `on_agent_finish` — called when the agent produces its final answer
- `on_tool_start` — called just before tool execution begins
- `on_tool_end` — called with the tool's output

```python
result = agent.invoke(
    {"input": "..."},
    config={"callbacks": [callback_handler]}
)
```

> **Callbacks vs Middleware:** Callbacks fire per-event and are useful for dev-time inspection and third-party integrations. For production flow control (deduplication, loop detection, metrics), prefer middleware — it has access to the full agent state at each step.

### 4.4 Structured Logging for Production

See [examples/AgentReasoning&Observability/4_ProductionLogging.py](examples/AgentReasoning&Observability/4_ProductionLogging.py) for a `ProductionAgentCallback` that writes structured log entries to a timestamped file.

```python
logging.basicConfig(
    filename=f'agent_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

> **Pro tip:** For zero-code full tracing (LLM calls, token usage, latencies), set `LANGCHAIN_TRACING_V2=true` to enable LangSmith.

---

## 5. Multi-Agent Architectures

→ Examples: [examples/MultiAgent&Orchestration/](examples/MultiAgent&Orchestration/)

### 5.1 Pattern 1: Planner + Worker

See [examples/MultiAgent&Orchestration/1_PlannerWorker.py](examples/MultiAgent&Orchestration/1_PlannerWorker.py).

One component (an LCEL chain) plans; another (a `create_agent`) executes. The Planner returns a numbered list; the Worker receives the full plan as its task:

```
STEP 1: PLANNING  →  Planner chain produces numbered plan
STEP 2: EXECUTION →  Worker agent executes the plan with tools
```

The Planner is a pure LCEL chain (no agent needed for text-only planning). The Worker uses `fetch_data`, `process_data`, and `generate_report` tools.

### 5.2 Pattern 2: Researcher + Writer

See [examples/MultiAgent&Orchestration/2_ResearcherWriter.py](examples/MultiAgent&Orchestration/2_ResearcherWriter.py).

Domain-separated agents with different tool sets:
- **Researcher** tools: `search_papers`, `search_news`, `search_statistics`
- **Writer** tools: `create_outline`, `format_article`

```
PHASE 1: RESEARCH  →  Researcher gathers findings
PHASE 2: WRITING   →  Writer receives findings and produces article
```

### 5.3 Pattern 3: Critic + Builder

See [examples/MultiAgent&Orchestration/3_CriticBuilder.py](examples/MultiAgent&Orchestration/3_CriticBuilder.py).

An iterative feedback loop using two LCEL chains:
- **Builder** generates code from requirements (or revises based on feedback)
- **Critic** scores the code and either approves (`APPROVED: <reason>`) or requests changes (`NEEDS_WORK: <issues>`)

The loop continues until approved or `max_iterations` is reached:

```python
result = critic_builder_loop(requirements, max_iterations=3)
# result["status"] → "approved" or "max_iterations_reached"
```

### 5.4 Pattern 4: Manager + Workers (Hierarchical)

See [examples/MultiAgent&Orchestration/4_ManagerWorker.py](examples/MultiAgent&Orchestration/4_ManagerWorker.py) for a `ManagerWorkerSystem` that implements the hierarchical pattern.

```
        Manager
       /   |   \
Worker1 Worker2 Worker3
```

The Manager LCEL chain parses a high-level task into `ASSIGN: worker_name | task_description` directives and dispatches subtasks to named Worker agents.

### 5.5 Communication Patterns

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

**Pattern C: Iterative (Feedback Loop)**
```
Builder → Critic → (approved?) → done
              ↓ (needs work)
           Builder (revises)
```

**Passing output between agents:**

```python
result1 = agent1.invoke({"messages": [{"role": "user", "content": task}]})
output  = result1["messages"][-1].content
result2 = agent2.invoke({"messages": [{"role": "user", "content": output}]})
```

---

## 6. Production Debugging & Pitfalls

→ Examples: [examples/ProductionPatterns/](examples/ProductionPatterns/)

### 6.1 Issue 1: Over-invocation of Tools

See [examples/ProductionPatterns/1_OverInvocation.py](examples/ProductionPatterns/1_OverInvocation.py).

**Problem:** Agent calls the same tool repeatedly with similar inputs.

**Solutions:**

1. **Improve tool description** — add `"IMPORTANT: Only call this once per unique query."` to the docstring
2. **`DeduplicationMiddleware`** — tracks `(tool, input)` pairs and raises if a duplicate is seen
3. **System prompt reinforcement** — explicit rules in `system_prompt`

### 6.2 Issue 2: Runaway Loops

See [examples/ProductionPatterns/2_RunawayLoops.py](examples/ProductionPatterns/2_RunawayLoops.py).

**Problem:** Agent enters a loop without making progress when tools consistently return no results.

**Solutions:**

1. **`LoopGuardMiddleware`** — detects when the same tool observation repeats and raises
2. **Enhanced system prompt** — explicit rules: "If a tool returns 'No results', DO NOT retry the same query"
3. **Async timeout wrapper** — `asyncio.wait_for` for wall-clock limits

### 6.3 Issue 3: Parsing Errors

See [examples/ProductionPatterns/3_ParsingErrors.py](examples/ProductionPatterns/3_ParsingErrors.py).

**Problem:** Text-parsed agents raise `OutputParserException` when the LLM produces malformed output.

**Solution:** `create_agent` uses the model's native tool-calling API (structured JSON). There is no freeform text to parse, so parsing errors are eliminated by design. For open-source LLMs without tool-calling support, LangGraph's `create_react_agent` with `handle_parsing_errors` is the fallback (covered in the LangGraph session).

### 6.4 Issue 4: Tool Execution Errors

See [examples/ProductionPatterns/4_ToolErrors.py](examples/ProductionPatterns/4_ToolErrors.py).

**Problem:** Tools crash (e.g., `ZeroDivisionError`) or stall (external API timeouts).

**Solutions:**

1. **`safe_calculator`** — per-exception-type error messages returned as strings
2. **`validated_calculator`** — pre-validate the expression before evaluating
3. **`run_with_timeout`** — cross-platform timeout via `concurrent.futures.ThreadPoolExecutor`

> **Cross-platform note:** `signal.SIGALRM` is Unix-only. `concurrent.futures` works on Windows, Linux, and macOS.

**Middleware hooks available on `BaseMiddleware`:**

| Hook | When it fires |
|------|---------------|
| `before_model(state, config)` | Before each LLM call |
| `after_model(state, config)` | After each LLM call |
| `after_tool(state, config)` | After each tool execution |
| `on_error(error, state, config)` | On any exception in the loop |

Each hook receives the full agent state dict and must return `(state, config)`.

### 6.5 Monitoring and Observability

See [examples/ProductionPatterns/5_Monitoring.py](examples/ProductionPatterns/5_Monitoring.py) for `AgentMonitor` + `MonitoringMiddleware` tracking:
- Total calls, success rate, average execution time
- Per-tool invocation counts
- Last 5 error records with timestamps

```python
monitor = AgentMonitor()
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[MonitoringMiddleware(monitor)],
)

# After running queries...
print(monitor.get_report())
```

### 6.6 Guardrails and Constraints

See [examples/ProductionPatterns/6_Guardrails.py](examples/ProductionPatterns/6_Guardrails.py) for a `GuardrailAgent` wrapper with:
- **Pre-execution input checks:** max query length, prohibited term list
- **Post-execution output checks:** PII detection (email/phone regex patterns)

```python
rules = {
    'max_query_length': 500,
    'prohibited_terms': ['hack', 'exploit', 'bypass'],
    'allow_pii': False
}
guarded_agent = GuardrailAgent(base_agent, rules)
```

---

## 7. Advanced Capstone Project

→ Example: [examples/CompleteKnowledgeWorkerSystem.py](examples/CompleteKnowledgeWorkerSystem.py)

### Multi-Agent Knowledge Worker System

**Scenario:** A complete three-agent pipeline that researches, writes, and evaluates documentation in an iterative quality loop.

**Agents:**
1. **Researcher** (`create_agent`) — gathers information using `search_web`, `search_documentation`, `search_examples` tools
2. **Writer** (`create_agent`) — creates structured markdown documentation using `create_document_structure`, `format_markdown`, `add_code_examples` tools
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
# Fill in keys/.env first
python examples/CompleteKnowledgeWorkerSystem.py
```

### Extension Ideas

1. **Add More Agents:** SEO Optimizer, Code Tester, Translator
2. **Enhanced Communication:** Shared memory between agents, parallel execution
3. **Production Features:** Database persistence, API endpoints, real-time progress, human-in-the-loop approval

---

## 8. References & Resources

### Official Documentation

- **LangChain Concepts**: https://python.langchain.com/docs/concepts/
- **Agents Guide**: https://python.langchain.com/docs/how_to/#agents
- **Tools Guide**: https://python.langchain.com/docs/how_to/#tools
- **LCEL**: https://python.langchain.com/docs/concepts/lcel/
- **LangSmith**: https://docs.smith.langchain.com/

### Best Practices

1. **Start Simple:** Begin with a single agent, add complexity gradually
2. **Write Clear Tool Descriptions:** The agent's tool selection is only as good as your descriptions
3. **Monitor Everything:** Use middleware and logging, or enable LangSmith tracing
4. **Set Limits:** Always configure max iterations and timeouts to prevent runaway costs
5. **Handle Errors Gracefully:** Tools should return readable error strings, never raise
6. **Test Thoroughly:** Agents are non-deterministic — test edge cases
7. **Build Incrementally:** Build → Test → Refine rather than designing the full system up front

### Common Pitfalls

❌ **Over-relying on agents** — not everything needs an agent  
❌ **Vague tool descriptions** — leads to poor tool selection  
❌ **No iteration limits** — can cause runaway costs  
❌ **Ignoring tool errors** — tools should always return strings, even on failure  
❌ **Missing monitoring** — you can't debug what you can't observe  
❌ **Starting with complex multi-agent** — build incrementally  
❌ **No human oversight** — agents make mistakes; have review processes  
❌ **Using `eval()` in tools** — use `numexpr` for safe math evaluation  

### Production Checklist

- [ ] Error handling in all tools (return strings, never raise)
- [ ] Iteration and time limits configured
- [ ] Logging and monitoring in place
- [ ] Deduplication and loop prevention middleware
- [ ] Cost tracking and alerts
- [ ] Security guardrails (input validation, output PII filtering)
- [ ] Output validation
- [ ] Fallback mechanisms
- [ ] Human approval gates for critical actions
- [ ] Pinned package versions in `requirements.txt`
- [ ] LangSmith tracing enabled (or equivalent)

### Next Steps

- **LangGraph**: Modern state-machine orchestration for multi-agent systems (next session)
- **Memory Systems**: Long-term agent memory via `RunnableWithMessageHistory`
- **Human-in-the-Loop**: Interactive agent workflows requiring human approval
- **Distributed Agents**: Scaling across multiple instances

---

## Project Structure

```
.
├── requirements.txt
├── keys/
│   └── .env                                     # Azure OpenAI credentials (never commit)
└── examples/
    ├── Foundations/
    │   ├── 1_LCELChain.py                       # Deterministic LCEL chain
    │   └── 2_ReactAgent.py                      # Basic agent with tool selection
    ├── AgentTools/
    │   ├── 1_ToolDecorator.py                   # @tool decorator, best practices
    │   ├── 2_ToolClass.py                       # Tool class constructor
    │   ├── 3_StructuredTool.py                  # StructuredTool with Pydantic schema
    │   └── 4_BuiltinTools.py                    # Calculator, search, retrieval
    ├── SingleAgent/
    │   ├── 1_FirstAgent.py                      # Complete first agent
    │   ├── 2_AgentTypes.py                      # Standard vs specialist agent
    │   └── 3_AgentConfig.py                     # create_agent parameters reference
    ├── AgentReasoning&Observability/
    │   ├── 1_VerboseMode.py                     # Reading the reasoning trace
    │   ├── 2_IntermediateSteps.py               # Extracting tool call/result pairs
    │   ├── 3_CustomCallbacks.py                 # BaseCallbackHandler for event hooks
    │   └── 4_ProductionLogging.py               # Structured file logging
    ├── MultiAgent&Orchestration/
    │   ├── 1_PlannerWorker.py                   # Planner chain + Worker agent
    │   ├── 2_ResearcherWriter.py                # Domain-separated sequential agents
    │   ├── 3_CriticBuilder.py                   # Iterative refinement loop
    │   └── 4_ManagerWorker.py                   # Hierarchical dispatch
    ├── ProductionPatterns/
    │   ├── 1_OverInvocation.py                  # DeduplicationMiddleware
    │   ├── 2_RunawayLoops.py                    # LoopGuardMiddleware + async timeout
    │   ├── 3_ParsingErrors.py                   # Why tool-calling eliminates parse errors
    │   ├── 4_ToolErrors.py                      # Safe error handling and timeouts
    │   ├── 5_Monitoring.py                      # AgentMonitor + MonitoringMiddleware
    │   └── 6_Guardrails.py                      # GuardrailAgent with PII filtering
    └── CompleteKnowledgeWorkerSystem.py         # Capstone: full 3-agent pipeline
```

---

## Conclusion

This guide covers the full arc from fundamentals to production:

- **Foundations** — when to use a chain vs an agent, how the ReAct loop works, and how tool descriptions drive selection
- **Tools** — three creation methods (`@tool`, `Tool`, `StructuredTool`), best practices for descriptions and error handling, and built-in integrations
- **Single Agents** — the `create_agent` pattern, customizing behavior via `system_prompt`, reading the reasoning trace, and the full parameter reference
- **Observability** — reading `result["messages"]`, extracting intermediate steps, callbacks for dev inspection, and structured file logging
- **Multi-Agent Patterns** — sequential pipeline, domain-separated agents, iterative refinement loop, and hierarchical dispatch
- **Production Patterns** — over-invocation, runaway loops, parsing errors, tool execution failures, metrics middleware, and safety guardrails
- **Capstone** — a full three-agent system combining research, writing, and iterative evaluation

**Next session: LangGraph** — state-machine orchestration, persistent agent state, and parallel branches.
