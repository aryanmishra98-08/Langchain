# LangChain Fundamentals: A Progressive Learning Guide

A structured, example-driven repository for learning [LangChain](https://www.langchain.com/) — the leading open-source framework for building applications powered by large language models. Each example builds on the last, taking you from your first LLM call to a production-ready conversational chatbot.

**License:** Apache 2.0

---

## Philosophy

LangChain is powerful, but its breadth can be overwhelming for newcomers. This repository distills the framework into **21 focused, runnable examples** organized into four progressive learning tracks. The guiding principles are:

- **Learn by doing.** Every concept is a standalone Python script you can run, modify, and experiment with immediately.
- **Progressive complexity.** Examples are numbered sequentially — each one introduces exactly one new idea on top of what you already know.
- **Production awareness.** The journey doesn't stop at "hello world." The final examples cover persistent storage, session management, and memory optimization patterns used in real deployments.
- **Modern idioms.** All code uses the current LangChain API surface (`langchain-core`, LCEL, `RunnableWithMessageHistory`) rather than deprecated legacy classes.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Structure](#repository-structure)
3. [What is LangChain?](#what-is-langchain)
4. [Track 1 — Setup and Core Interactions](#track-1--setup-and-core-interactions-examples-14)
5. [Track 2 — Prompt Engineering](#track-2--prompt-engineering-examples-57)
6. [Track 3 — Chains with LCEL](#track-3--chains-with-lcel-examples-814)
7. [Track 4 — Conversational Memory](#track-4--conversational-memory-examples-1521)
8. [Core Concepts at a Glance](#core-concepts-at-a-glance)
9. [Navigating the Examples](#navigating-the-examples)
10. [Memory Storage Comparison](#memory-storage-comparison)
11. [Migrating from Legacy APIs](#migrating-from-legacy-apis)
12. [Additional Resources](#additional-resources)

---

## Quick Start

### Prerequisites

- Python 3.9 or later
- An API key from a supported LLM provider (Azure OpenAI, OpenAI, or Anthropic)

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd Langchain
pip install -r requirement.txt
```

### 2. Configure your API key

Create a `keys/.env` file with your credentials:

```env
# Azure OpenAI (used by default in the examples)
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

All examples load this file automatically via:

```python
from dotenv import load_dotenv
load_dotenv("keys/.env")
```

### 3. Run your first example

```bash
python "examples/1 Test Install.py"
```

If you see `LangChain installed successfully!`, you're ready to go.

---

## Repository Structure

```
Langchain/
├── README.md                 ← You are here
├── requirement.txt           ← Python dependencies
├── LICENSE                   ← Apache 2.0
├── keys/
│   └── .env                  ← Your API keys (not committed)
└── examples/
    ├── 1 Test Install.py                         ← Verify setup
    ├── 2 First Call.py                           ← First LLM invocation
    ├── 3 Working with message.py                 ← Typed message roles
    ├── 4. LLM Streaming Response.py              ← Token-by-token streaming
    ├── 5 Prompt Template.py                      ← Variable prompt templates
    ├── 6 Chat Pompt Templates.py                 ← Multi-role chat templates
    ├── 7 Real World Example 1.py                 ← Q&A with context
    ├── 7 Real World Example 2.py                 ← Language translation
    ├── 7 Real World Example 3.py                 ← Text summarization
    ├── 7 Real World Example 4.py                 ← Code review assistant
    ├── 8 LCEL.py                                 ← Introduction to LCEL
    ├── 9 LCEL Flow.py                            ← Tracing chain execution
    ├── 10 Multiple Inputs LCEL.py                ← Multi-variable chains
    ├── 11 Chaining Multiple Operations.py        ← Sequential chain composition
    ├── 12 Using RunnablePassthrough.py           ← Passing context through chains
    ├── 13 Streaming Chains.py                    ← Streaming full LCEL chains
    ├── 14 Real World Example.py                  ← Concept explainer with code
    ├── 15 Conversation Memory.py                 ← In-memory chat history
    ├── 16 Multi-User Memory Support.py           ← Session-isolated histories
    ├── 17 Using SQLite Memory Support.py         ← Persistent SQLite storage
    ├── 18 Using Redis Memory Support.py          ← Distributed Redis storage
    ├── 19 Window Memory Pattern - N messages.py  ← Sliding window memory
    ├── 20 Summary Memory Pattern.py              ← Summarized long-term memory
    └── 21 Complete Production Example.py         ← Production chatbot
```

---

## What is LangChain?

**LangChain** is a framework for developing applications powered by large language models (LLMs). It simplifies LLM application development by providing modular components you can compose together instead of writing everything from scratch.

### Why use LangChain?

Without it, you would need to:
- Write custom code for every LLM API call and provider
- Manually manage conversation history and context
- Build complex prompt formatting logic from scratch
- Handle different LLM providers with different APIs

LangChain abstracts all of this into a unified, composable interface.

### Core building blocks

| Concept | Description |
|---------|-------------|
| **Models** | Unified interface to LLM providers (Azure OpenAI, OpenAI, Anthropic, etc.) |
| **Messages** | Typed conversation turns: `SystemMessage`, `HumanMessage`, `AIMessage` |
| **Prompt Templates** | Reusable, parameterized prompt structures |
| **Chains (LCEL)** | Declarative composition of operations using the `\|` pipe operator |
| **Output Parsers** | Extract and transform model responses into usable formats |
| **Memory** | Systems for maintaining conversation context across turns |

---

## Track 1 — Setup and Core Interactions (Examples 1–4)

### Theory

At its core, LangChain wraps LLM provider APIs into a consistent interface. The two key invocation methods are:

- **`invoke(input)`** — Send a prompt and get a complete response back.
- **`stream(input)`** — Send a prompt and receive the response token-by-token as it is generated.

Modern LLMs use a **chat model** interface, which distinguishes between message roles:

| Role | Class | Purpose |
|------|-------|---------|
| `system` | `SystemMessage` | Sets the model's persona or behavioral rules |
| `human` | `HumanMessage` | Represents the user's input |
| `ai` | `AIMessage` | Represents the model's previous responses |

The model reads all messages in order and generates the next response. You control behavior primarily through the `system` message.

The `temperature` parameter controls output randomness:
- `temperature=0` → deterministic, consistent answers (best for factual tasks)
- `temperature=1` → more creative and varied outputs (best for writing tasks)

---

### Example 1 — Verify Installation

**File:** `examples/1 Test Install.py`

Confirms that `langchain-openai`, your Azure credentials, and Python are all correctly configured.

```python
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv("keys/.env")

llm = AzureChatOpenAI(model="gpt-5.1-chat", temperature=0.9)
print("LangChain installed successfully!")
```

**Run it:**
```bash
python "examples/1 Test Install.py"
```

---

### Example 2 — First LLM Call

**File:** `examples/2 First Call.py`

Makes a real call to the model and prints the response.

```python
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv("keys/.env")

llm = AzureChatOpenAI(model="gpt-5.1-chat", temperature=1)

response = llm.invoke("What is LangChain?")
print(response.content)
```

- `llm.invoke("...")` accepts a plain string and returns an `AIMessage` object.
- `.content` extracts the plain text string from that object.

**Run it:**
```bash
python "examples/2 First Call.py"
```

---

### Example 3 — Working with Messages

**File:** `examples/3 Working with message.py`

Demonstrates passing a structured list of messages rather than a plain string.

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv("keys/.env")

llm = AzureChatOpenAI(model="gpt-5.1-chat")

messages = [
    SystemMessage(content="You are a helpful Python tutor."),
    HumanMessage(content="What is a list comprehension?"),
]

response = llm.invoke(messages)
print(response.content)
```

The `SystemMessage` shapes how the model behaves — here it responds as a Python tutor. This is the foundation of prompt engineering: controlling model persona and behavior through system instructions.

**Run it:**
```bash
python "examples/3 Working with message.py"
```

---

### Example 4 — Streaming Responses

**File:** `examples/4. LLM Streaming Response.py`

Streams the response token-by-token instead of waiting for the full reply.

```python
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv("keys/.env")

llm = AzureChatOpenAI(model="gpt-5.1-chat")

for chunk in llm.stream("Write a short poem about Python programming"):
    print(chunk.content, end="", flush=True)
```

- `llm.stream(...)` returns an iterator of response chunks.
- `end=""` prevents a newline after each chunk so the output flows continuously.
- `flush=True` forces each chunk to print immediately without buffering.

Streaming is essential for chatbots and any UI that should feel responsive rather than frozen while waiting for a long response.

**Run it:**
```bash
python "examples/4. LLM Streaming Response.py"
```

---

## Track 2 — Prompt Engineering (Examples 5–7)

### Theory

A **Prompt Template** is a reusable prompt structure with named placeholder variables. Instead of constructing prompt strings manually, you define the structure once and fill in variables at runtime.

**Why this matters:**
- Your prompt logic is separated from your application logic.
- Templates can be tested, versioned, and reused across the codebase.
- Variables make prompts dynamic and configurable without string concatenation.

There are two main template types:

| Class | Use When |
|-------|----------|
| `PromptTemplate` | You have a single text block with variables |
| `ChatPromptTemplate` | You need multi-role messages (system + human) with variables |

**`PromptTemplate` — single text block:**

```python
template = "Tell me a {adjective} joke about {topic}."
prompt = PromptTemplate(input_variables=["adjective", "topic"], template=template)
formatted = prompt.format(adjective="funny", topic="programming")
# → "Tell me a funny joke about programming."
```

**`ChatPromptTemplate` — multi-role messages:**

```python
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {profession} assistant."),
    ("human", "Help me with: {task}")
])
messages = chat_template.format_messages(profession="Python", task="decorators")
# → [SystemMessage(...), HumanMessage(...)]
```

---

### Example 5 — Prompt Template

**File:** `examples/5 Prompt Template.py`

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv("keys/.env")

template = "Tell me a {adjective} joke about {topic}."

prompt = PromptTemplate(
    input_variables=["adjective", "topic"],
    template=template
)

formatted_prompt = prompt.format(adjective="funny", topic="programming")
print(formatted_prompt)  # Tell me a funny joke about programming.

llm = AzureChatOpenAI(model="gpt-5.1-chat")
response = llm.invoke(formatted_prompt)
print(response.content)
```

**Run it:**
```bash
python "examples/5 Prompt Template.py"
```

---

### Example 6 — Chat Prompt Templates

**File:** `examples/6 Chat Pompt Templates.py`

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv("keys/.env")

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {profession} assistant."),
    ("human", "Help me with: {task} and believe me, I need it badly for {reason}")
])

formatted = chat_template.format_messages(
    profession="Python programming",
    task="understanding decorators",
    reason="an upcoming project deadline"
)

llm = AzureChatOpenAI(model="gpt-5.1-chat")
response = llm.invoke(formatted)
print(response.content)
```

Variables can appear in both the system and human messages. `format_messages()` returns a list of fully formatted `BaseMessage` objects, ready to pass directly to the model.

**Run it:**
```bash
python "examples/6 Chat Pompt Templates.py"
```

---

### Examples 7a–7d — Real-World Prompt Patterns

These four examples apply prompt templates to the most common production use cases. Each one demonstrates a different template structure to solve a distinct problem.

#### 7a — Q&A with Context (`7 Real World Example 1.py`)

Grounds the model's answer in a provided context block. When the answer is not in the context, the model says "I don't know" rather than hallucinating.

```python
qa_template = """
Use the following context to answer the question.
If you cannot answer based on the context, say "I don't know."

Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=qa_template)

context = "LangChain is a framework for building LLM applications. It was created in 2022."
formatted = prompt.format(context=context, question="When was LangChain created?")

llm = AzureChatOpenAI(model="gpt-5.1-chat")
response = llm.invoke(formatted)
print(response.content)
# → LangChain was created in 2022.
```

**Run it:** `python "examples/7 Real World Example 1.py"`

---

#### 7b — Language Translation (`7 Real World Example 2.py`)

Uses the system message to establish a translator persona, then passes source language, target language, and text as variables.

```python
translation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Translate from {source_lang} to {target_lang}."),
    ("human", "{text}")
])

messages = translation_template.format_messages(
    source_lang="English",
    target_lang="Spanish",
    text="Hello, how are you?"
)

response = llm.invoke(messages)
print(response.content)
# → Hola, ¿cómo estás?
```

**Run it:** `python "examples/7 Real World Example 2.py"`

---

#### 7c — Text Summarization (`7 Real World Example 3.py`)

Summarizes a body of text into a configurable number of sentences.

```python
summary_template = """
Summarize the following text in {num_sentences} sentences:

{text}

Summary:
"""

prompt = PromptTemplate(input_variables=["text", "num_sentences"], template=summary_template)
formatted = prompt.format(text=long_text, num_sentences=2)
response = llm.invoke(formatted)
print(response.content)
```

**Run it:** `python "examples/7 Real World Example 3.py"`

---

#### 7d — Code Review (`7 Real World Example 4.py`)

Reviews a code snippet for a given programming language, providing feedback on issues, improvements, and best practices.

```python
code_review_template = """
You are a senior software engineer performing a code review for the following language: {language}.
Review the following code and provide feedback on potential issues, improvements, and best practices.
{code}
Feedback:
"""

prompt = PromptTemplate(input_variables=["language", "code"], template=code_review_template)
formatted = prompt.format(language="Python", code=code_snippet)
response = llm.invoke(formatted)
print(response.content)
```

**Run it:** `python "examples/7 Real World Example 4.py"`

---

## Track 3 — Chains with LCEL (Examples 8–14)

### Theory

**LCEL (LangChain Expression Language)** is a declarative syntax for composing chains of operations using the `|` pipe operator. It is the modern, recommended way to build LangChain applications.

The fundamental pattern is:

```
prompt | model | output_parser
```

Each component is a **Runnable** — an object with a consistent interface (`invoke`, `stream`, `batch`). Piping Runnables together creates a new Runnable that runs each step in sequence, passing the output of one as the input to the next.

**Why LCEL over manual composition?**
- Clean, readable syntax that mirrors Unix pipes.
- The entire chain automatically inherits `.stream()`, `.invoke()`, and `.batch()`.
- Easy to inspect individual steps for debugging.
- Built-in support for async and parallel execution.

**Step-by-step data flow through a chain:**

```
{"input": "What is 2+2?"}
        ↓  prompt
ChatPromptValue([SystemMessage, HumanMessage])
        ↓  model
AIMessage(content="4")
        ↓  StrOutputParser
"4"
```

**`StrOutputParser`** extracts the plain text string from the model's `AIMessage` response, so the final output is a `str` rather than a message object.

**`RunnablePassthrough`** is a special Runnable that forwards its input unchanged. It is used when you need to pass a value from the chain's input dict directly into a downstream prompt variable without any transformation.

---

### Example 8 — Your First LCEL Chain

**File:** `examples/8 LCEL.py`

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv("keys/.env")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

model = AzureChatOpenAI(model="gpt-5.1-chat")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

result = chain.invoke({"input": "What is 2+2?"})
print(result)
```

**Run it:**
```bash
python "examples/8 LCEL.py"
```

---

### Example 9 — Tracing Chain Execution

**File:** `examples/9 LCEL Flow.py`

Shows exactly what happens at each step by invoking each component individually before running the whole chain at once.

```python
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = AzureChatOpenAI(model="gpt-5.1-chat")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Trace step-by-step
input_data = {"topic": "Python"}

formatted = prompt.invoke(input_data)
print(formatted)                                      # ChatPromptValue

model_output = model.invoke(formatted)
print(type(model_output), model_output.content[:50])  # AIMessage

final_output = output_parser.invoke(model_output)
print(final_output)                                   # plain string

# Or run in one shot:
print(chain.invoke({"topic": "Python"}))
```

Use this approach when debugging unexpected chain behaviour — step through each component to pinpoint where output diverges from expectation.

**Run it:**
```bash
python "examples/9 LCEL Flow.py"
```

---

### Example 10 — Multiple Inputs

**File:** `examples/10 Multiple Inputs LCEL.py`

Passes multiple variables to a single chain in one `invoke()` call.

```python
prompt = ChatPromptTemplate.from_template(
    "Translate the following {source_lang} text to {target_lang}: {text}"
)

chain = prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

result = chain.invoke({
    "source_lang": "English",
    "target_lang": "French",
    "text": "Good morning"
})
print(result)  # Bonjour
```

**Run it:**
```bash
python "examples/10 Multiple Inputs LCEL.py"
```

---

### Example 11 — Chaining Multiple Operations

**File:** `examples/11 Chaining Multiple Operations.py`

Composes two separate chains sequentially, where the output of the first chain becomes the input of the second.

```python
# Chain 1: Generate a topic
topic_prompt = ChatPromptTemplate.from_template("Suggest a random topic related to {subject}")
topic_chain = topic_prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

# Chain 2: Write about that topic
writing_prompt = ChatPromptTemplate.from_template("Write a short paragraph about: {topic}")
writing_chain = writing_prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

# Run sequentially
topic = topic_chain.invoke({"subject": "artificial intelligence"})
print(f"Generated topic: {topic}\n")

paragraph = writing_chain.invoke({"topic": topic})
print(f"Paragraph:\n{paragraph}")
```

This pattern is the basis for **pipeline architectures** where multiple LLM calls build upon each other.

**Run it:**
```bash
python "examples/11 Chaining Multiple Operations.py"
```

---

### Example 12 — RunnablePassthrough

**File:** `examples/12 Using RunnablePassthrough.py`

Uses `RunnablePassthrough` to route multiple values from the input dict directly to the prompt without transformation.

```python
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below:

Context: {context}
Question: {question}
Answer:
""")

chain = (
    {
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough()
    }
    | prompt
    | AzureChatOpenAI(model="gpt-5.1-chat")
    | StrOutputParser()
)

result = chain.invoke({
    "context": "The sky appears blue due to Rayleigh scattering.",
    "question": "Why is the sky blue?"
})
print(result)
```

The dict `{"context": RunnablePassthrough(), "question": RunnablePassthrough()}` is itself a Runnable that maps each input key to its value, passing them through unchanged to the prompt template.

**Run it:**
```bash
python "examples/12 Using RunnablePassthrough.py"
```

---

### Example 13 — Streaming Chains

**File:** `examples/13 Streaming Chains.py`

The same `stream()` method that works on a bare model works identically on any LCEL chain.

```python
chain = prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

for chunk in chain.stream({"topic": "a robot learning to paint"}):
    print(chunk, end="", flush=True)
```

Because `StrOutputParser` is in the chain, each `chunk` is already a plain string rather than an `AIMessage` — you get clean, directly printable output.

**Run it:**
```bash
python "examples/13 Streaming Chains.py"
```

---

### Example 14 — Real-World LCEL: Concept Explainer

**File:** `examples/14 Real World Example.py`

A practical chain that explains a programming concept with a code example. Also demonstrates using both `invoke` and `stream` on the same chain.

```python
prompt = ChatPromptTemplate.from_template("""
Explain the following concept using a code example: {concept} in the following language: {language}
""")

chain = (
    {"concept": RunnablePassthrough(), "language": RunnablePassthrough()}
    | prompt
    | AzureChatOpenAI(model="gpt-5.1-chat")
    | StrOutputParser()
)

concept = "chaining multiple operations in Langchain"
language = "Python"

# Full response at once
result = chain.invoke({"concept": concept, "language": language})
print(result)

# Or stream it token by token
for chunk in chain.stream({"concept": concept, "language": language}):
    print(chunk, end="", flush=True)
```

**Run it:**
```bash
python "examples/14 Real World Example.py"
```

---

## Track 4 — Conversational Memory (Examples 15–21)

### Theory

Without memory, every LLM call is completely stateless — the model has no knowledge of previous turns:

```
# Without memory
User: My name is Alice
AI:   Nice to meet you!
User: What's my name?
AI:   I don't know your name.   ← no memory
```

With memory, the full conversation history is included in every prompt:

```
# With memory
User: My name is Alice
AI:   Nice to meet you, Alice!
User: What's my name?
AI:   Your name is Alice!       ← remembers
```

#### How the modern memory pattern works

The approach has three parts:

1. **`MessagesPlaceholder` in the prompt** — reserves a slot where conversation history will be automatically injected before each LLM call.
2. **A history store** — a dictionary or database mapping `session_id` strings to `ChatMessageHistory` objects.
3. **`RunnableWithMessageHistory`** — wraps any LCEL chain. Before each `invoke()`, it loads the history for the given session ID and injects it at the `MessagesPlaceholder`. After the response, it saves the new messages back to the store.

```
User sends message
        ↓
RunnableWithMessageHistory loads history for session_id
        ↓
History injected at MessagesPlaceholder in prompt
        ↓
Chain runs: prompt | model | parser
        ↓
Response returned to user
        ↓
New HumanMessage + AIMessage saved back to history store
```

#### Session IDs

A **session ID** is a string that uniquely identifies one conversation. Different session IDs have completely independent histories — this is how multi-user support works.

```python
config = {"configurable": {"session_id": "user_alice_001"}}
response = chain_with_history.invoke({"input": "Hello"}, config=config)
```

#### Memory components reference

| Component | Purpose |
|-----------|---------|
| `MessagesPlaceholder` | Placeholder in prompt for chat history |
| `InMemoryChatMessageHistory` | Stores messages in RAM (development/testing) |
| `RunnableWithMessageHistory` | Wraps a chain to manage history automatically |
| `SQLChatMessageHistory` | Persistent storage via SQLite or PostgreSQL |
| `RedisChatMessageHistory` | Fast distributed storage with TTL support |

---

### Example 15 — Basic Conversation Memory

**File:** `examples/15 Conversation Memory.py`

The complete four-step pattern for adding memory to any LCEL chain.

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv("keys/.env")

# Step 1: Prompt with MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI assistant. Remember what users tell you."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Step 2: Build the chain
model = AzureChatOpenAI(model="gpt-5.1-chat")
chain = prompt | model | StrOutputParser()

# Step 3: History store and factory function
conversation_store = {}

def get_chat_history(session_id: str):
    if session_id not in conversation_store:
        conversation_store[session_id] = InMemoryChatMessageHistory()
    return conversation_store[session_id]

# Step 4: Wrap the chain
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="question",      # matches the prompt variable
    history_messages_key="chat_history" # matches the MessagesPlaceholder name
)

# Use it
session_config = {"configurable": {"session_id": "session_001"}}

response = conversational_chain.invoke({"question": "Hi, I'm learning LangChain!"}, config=session_config)
print(f"AI: {response}")

response = conversational_chain.invoke({"question": "What am I learning?"}, config=session_config)
print(f"AI: {response}")  # → "You're learning LangChain!"

response = conversational_chain.invoke({"question": "Can you quiz me on it?"}, config=session_config)
print(f"AI: {response}")
```

**Key parameters in `RunnableWithMessageHistory`:**
- `input_messages_key` — the key in your `invoke()` dict that holds the user's message. Must match the variable name in the prompt.
- `history_messages_key` — the key that matches the `MessagesPlaceholder` variable name in the prompt.

**Run it:**
```bash
python "examples/15 Conversation Memory.py"
```

---

### Example 16 — Multi-User Session Management

**File:** `examples/16 Multi-User Memory Support.py`

Demonstrates that different session IDs produce completely isolated conversation histories, enabling multi-user support with a single chain instance.

```python
chain_with_history = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Alice's session
user1_config = {"configurable": {"session_id": "alice_123"}}
chain_with_history.invoke({"input": "My name is Alice"}, config=user1_config)

# Bob's separate session
user2_config = {"configurable": {"session_id": "bob_456"}}
chain_with_history.invoke({"input": "My name is Bob"}, config=user2_config)

# Each user only sees their own history
chain_with_history.invoke({"input": "What's my name?"}, config=user1_config)
# → "Your name is Alice"

chain_with_history.invoke({"input": "What's my name?"}, config=user2_config)
# → "Your name is Bob"
```

**Production tip:** Use meaningful, collision-resistant session IDs:
```python
session_id = f"user_{user_id}_conv_{conversation_id}"
```

**Run it:**
```bash
python "examples/16 Multi-User Memory Support.py"
```

---

### Example 17 — SQLite Persistent Memory

**File:** `examples/17 Using SQLite Memory Support.py`

Replaces the in-memory store with a SQLite database so conversation history survives application restarts. The only change from Example 15 is the `get_sql_history` factory function — the chain and wrapper are identical.

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sql_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"  # file created automatically
    )

chain_with_history = RunnableWithMessageHistory(
    chain, get_sql_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "persistent_session"}}

chain_with_history.invoke({"input": "Remember: my favorite color is blue"}, config=config)

# Restart the program — history is still there:
chain_with_history.invoke({"input": "What's my favorite color?"}, config=config)
# → "Your favorite color is blue!"
```

**Run it:**
```bash
python "examples/17 Using SQLite Memory Support.py"
```

---

### Example 18 — Redis Persistent Memory

**File:** `examples/18 Using Redis Memory Support.py`

Uses Redis for fast, distributed storage suitable for high-traffic production deployments. The `ttl` parameter automatically expires old sessions.

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=86400   # session expires after 24 hours
    )

chain_with_history = RunnableWithMessageHistory(
    chain, get_redis_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "redis_session"}}
response = chain_with_history.invoke({"input": "Store this in Redis!"}, config=config)
print(response)
```

Redis requires a running server. For local development:
```bash
docker run -p 6379:6379 redis
```

**Other supported persistent backends:**
- `PostgresChatMessageHistory` — PostgreSQL
- `MongoDBChatMessageHistory` — MongoDB
- `DynamoDBChatMessageHistory` — AWS DynamoDB

**Run it:**
```bash
python "examples/18 Using Redis Memory Support.py"
```

---

### Example 19 — Window Memory Pattern

**File:** `examples/19 Window Memory Pattern - N messages.py`

A custom memory class that keeps only the last `k` messages, discarding older ones. This bounds token usage in long conversations.

```python
from typing import List
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

class WindowChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """Keeps only the last k messages."""
    messages: List[BaseMessage] = Field(default_factory=list)
    k: int = Field(default=10)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]  # trim to last k

    def clear(self) -> None:
        self.messages = []

# Use with k=4 (keep last 4 messages)
def get_window_history(session_id: str):
    if session_id not in window_store:
        window_store[session_id] = WindowChatMessageHistory(k=4)
    return window_store[session_id]
```

After sending five messages and asking about the first one, the model may not remember it — it has fallen outside the window. The most recent messages are always available.

**Trade-off:** The model cannot reference things said more than `k` messages ago. Useful when recent context is sufficient and you need predictable, bounded token costs.

**Run it:**
```bash
python "examples/19 Window Memory Pattern - N messages.py"
```

---

### Example 20 — Summary Memory Pattern

**File:** `examples/20 Summary Memory Pattern.py`

A more sophisticated pattern that summarizes older messages with a secondary LLM call instead of discarding them. The summary is stored as a `SystemMessage` prepended to the remaining recent messages, so the model retains the gist of the full conversation at a fraction of the token cost.

```python
class SummaryChatMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    llm: AzureChatOpenAI = Field(...)
    max_messages: int = Field(default=10)
    summary: str = Field(default="")

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

        if len(self.messages) > self.max_messages:
            # Summarize the first half
            to_summarize = self.messages[:self.max_messages // 2]
            conversation_text = "\n".join([
                f"{msg.__class__.__name__}: {msg.content}" for msg in to_summarize
            ])
            summary_prompt = (
                f"Previous summary: {self.summary}\n"
                f"New conversation:\n{conversation_text}\n"
                "Create a concise summary of the key information."
            )
            self.summary = self.llm.invoke(summary_prompt).content

            # Keep only recent messages, prepend summary
            self.messages = self.messages[self.max_messages // 2:]
            if self.summary:
                self.messages.insert(
                    0, SystemMessage(content=f"Summary of earlier conversation: {self.summary}")
                )

    def clear(self) -> None:
        self.messages = []
        self.summary = ""
```

**Trade-off:** Retains conceptual context across very long conversations but incurs an additional LLM call whenever the message count exceeds `max_messages`. Best for long-running sessions where topic continuity matters more than token cost.

**Run it:**
```bash
python "examples/20 Summary Memory Pattern.py"
```

---

### Example 21 — Production Chatbot (Capstone)

**File:** `examples/21 Complete Production Example.py`

Brings every concept together into a production-quality `ProductionChatbot` class. Features:

- SQLite-backed persistent history that survives restarts
- Dynamic `current_time` injection so the model is time-aware
- Error handling with graceful recovery messages
- Interactive REPL loop with `clear` and `exit` commands

```python
class ProductionChatbot:
    def __init__(self, db_path="chat_history.db"):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant.
            Current time: {current_time}
            Be concise and helpful!"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.model = AzureChatOpenAI(model="gpt-5.1-chat")
        self.chain = self.prompt | self.model | StrOutputParser()
        self.conversational_chain = RunnableWithMessageHistory(
            self.chain, self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def get_session_history(self, session_id: str):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{self.db_path}"
        )

    def chat(self, user_input: str, session_id: str):
        config = {"configurable": {"session_id": session_id}}
        try:
            return self.conversational_chain.invoke(
                {"input": user_input, "current_time": datetime.now().strftime("%Y-%m-%d %H:%M")},
                config=config
            )
        except Exception as e:
            return "I encountered an error. Please try again."

    def clear_history(self, session_id: str):
        self.get_session_history(session_id).clear()
```

**Run it:**
```bash
python "examples/21 Complete Production Example.py"
```

Type a message and press Enter. Type `clear` to reset the conversation history, or `exit` to quit.

---

## Core Concepts at a Glance

| Concept | What It Does | Where to Start |
|---------|-------------|----------------|
| **Models** | Interface to LLM providers (Azure OpenAI, OpenAI, Anthropic) | Example 2 |
| **Messages** | Typed conversation turns (`SystemMessage`, `HumanMessage`, `AIMessage`) | Example 3 |
| **Prompt Templates** | Reusable, parameterized prompt structures | Example 5 |
| **LCEL** | Declarative chain composition with the `\|` pipe operator | Example 8 |
| **Output Parsers** | Extract and format model responses (e.g., `StrOutputParser`) | Example 8 |
| **RunnablePassthrough** | Forward input values unchanged through a chain | Example 12 |
| **MessagesPlaceholder** | Reserve a slot in a prompt for injected conversation history | Example 15 |
| **RunnableWithMessageHistory** | Automatically load and save history around any chain | Example 15 |
| **Memory Backends** | Pluggable storage: in-memory, SQLite, Redis | Examples 15–18 |

---

## Navigating the Examples

**If you're brand new to LangChain:** Start at Example 1 and work through them in order. Each script is self-contained and can be run independently.

**If you know the basics and want to learn LCEL:** Jump to Example 8. Examples 8–14 form a complete LCEL mini-course.

**If you need to add memory to an existing app:** Start at Example 15 for the core pattern, then skip to whichever storage backend matches your needs (17 for SQLite, 18 for Redis).

**If you want a production reference:** Go directly to Example 21. It combines everything — LCEL chains, persistent storage, session management, and error handling — into a single, deployable chatbot class.

**Running any example:**
```bash
python "examples/<filename>.py"
```

All examples load environment variables from `keys/.env` automatically.

---

## Memory Storage Comparison

| Storage Backend | Best For | Persistence | Setup Complexity |
|----------------|----------|-------------|------------------|
| `InMemoryChatMessageHistory` | Development and testing | None (lost on restart) | None |
| `SQLChatMessageHistory` | Small-to-medium production apps | Disk (SQLite/PostgreSQL) | Low |
| `RedisChatMessageHistory` | High-traffic production apps | In-memory + optional disk | Medium (requires Redis) |
| Window Memory (custom) | Long conversations with bounded token cost | Depends on backing store | Low |
| Summary Memory (custom) | Very long conversations requiring full context | Depends on backing store | Medium |

---

## Migrating from Legacy APIs

If you have existing code using deprecated LangChain classes, here is the modern equivalent:

| Deprecated | Modern Replacement |
|-----------|-------------------|
| `ConversationChain` | LCEL chain (`prompt \| model \| parser`) |
| `ConversationBufferMemory` | `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` |
| `conversation.predict(input=...)` | `chain.invoke({"input": ...}, config=...)` |
| `LLMChain` | LCEL chain with `StrOutputParser` |

See Examples 15–21 for complete modern implementations of every memory pattern.

---

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [LangChain Expression Language (LCEL) Guide](https://python.langchain.com/docs/concepts/lcel/)
- [Chat History & Memory](https://python.langchain.com/docs/concepts/chat_history/)
- [LangSmith](https://smith.langchain.com/) — Observability and tracing for LangChain applications
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

---

<p align="center">
  Built for learning. Designed for production readiness.<br>
  Licensed under Apache 2.0.
</p>
