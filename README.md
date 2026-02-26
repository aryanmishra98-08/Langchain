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
3. [Learning Tracks](#learning-tracks)
   - [Track 1 — Setup and Core Interactions](#track-1--setup-and-core-interactions-examples-14)
   - [Track 2 — Prompt Engineering](#track-2--prompt-engineering-examples-57)
   - [Track 3 — Chains with LCEL](#track-3--chains-with-lcel-examples-814)
   - [Track 4 — Conversational Memory](#track-4--conversational-memory-examples-1521)
4. [Core Concepts at a Glance](#core-concepts-at-a-glance)
5. [Navigating the Examples](#navigating-the-examples)
6. [Memory Storage Comparison](#memory-storage-comparison)
7. [Migrating from Legacy APIs](#migrating-from-legacy-apis)
8. [Additional Resources](#additional-resources)

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

# Or standard OpenAI
OPENAI_API_KEY=your-key-here

# Or Anthropic Claude
ANTHROPIC_API_KEY=your-key-here
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

## Learning Tracks

### Track 1 — Setup and Core Interactions (Examples 1–4)

| Example | File | What You Learn |
|---------|------|----------------|
| 1 | `1 Test Install.py` | Verify that LangChain and your LLM provider are correctly configured |
| 2 | `2 First Call.py` | Initialize a chat model and make your first `invoke()` call |
| 3 | `3 Working with message.py` | Use `SystemMessage`, `HumanMessage`, and `AIMessage` to structure conversations |
| 4 | `4. LLM Streaming Response.py` | Stream responses token-by-token for real-time output |

**Key takeaway:** Chat models accept typed messages (system, human, AI) and return structured responses. The `invoke()` method sends a prompt; `stream()` yields results incrementally.

---

### Track 2 — Prompt Engineering (Examples 5–7)

| Example | File | What You Learn |
|---------|------|----------------|
| 5 | `5 Prompt Template.py` | Create reusable `PromptTemplate` objects with variable placeholders |
| 6 | `6 Chat Pompt Templates.py` | Build multi-role `ChatPromptTemplate` pipelines with system and human messages |
| 7a | `7 Real World Example 1.py` | **Pattern — Q&A:** Answer questions grounded in a provided context |
| 7b | `7 Real World Example 2.py` | **Pattern — Translation:** Translate between languages with role-based prompts |
| 7c | `7 Real World Example 3.py` | **Pattern — Summarization:** Condense text into a configurable number of sentences |
| 7d | `7 Real World Example 4.py` | **Pattern — Code Review:** Analyze code snippets for best practices |

**Key takeaway:** Prompt templates separate your application logic from prompt text. This makes prompts reusable, testable, and easy to iterate on. The four real-world patterns (7a–7d) demonstrate how templates apply to common production scenarios.

---

### Track 3 — Chains with LCEL (Examples 8–14)

| Example | File | What You Learn |
|---------|------|----------------|
| 8 | `8 LCEL.py` | The `prompt | model | parser` pattern — your first LCEL chain |
| 9 | `9 LCEL Flow.py` | Trace what happens at each stage of a chain's execution |
| 10 | `10 Multiple Inputs LCEL.py` | Pass multiple variables into a single chain |
| 11 | `11 Chaining Multiple Operations.py` | Compose sequential chains where one feeds into the next |
| 12 | `12 Using RunnablePassthrough.py` | Use `RunnablePassthrough` to forward data through complex pipelines |
| 13 | `13 Streaming Chains.py` | Stream output from a complete LCEL chain |
| 14 | `14 Real World Example.py` | **Pattern — Concept Explainer:** Explain a programming topic with code examples |

**Key takeaway:** LCEL (LangChain Expression Language) uses the `|` pipe operator to compose chains declaratively. The basic pattern is `prompt | model | output_parser`. LCEL chains automatically support streaming, async execution, and batch processing.

---

### Track 4 — Conversational Memory (Examples 15–21)

| Example | File | What You Learn |
|---------|------|----------------|
| 15 | `15 Conversation Memory.py` | Add in-memory conversation history with `RunnableWithMessageHistory` |
| 16 | `16 Multi-User Memory Support.py` | Isolate conversation state per user via session IDs |
| 17 | `17 Using SQLite Memory Support.py` | Persist history to SQLite so it survives application restarts |
| 18 | `18 Using Redis Memory Support.py` | Use Redis for fast, distributed history with TTL-based expiration |
| 19 | `19 Window Memory Pattern - N messages.py` | Keep only the last *N* messages to control token usage |
| 20 | `20 Summary Memory Pattern.py` | Summarize older messages while keeping recent ones in full |
| 21 | `21 Complete Production Example.py` | **Capstone:** A production chatbot class with SQL persistence, timestamps, error handling, and a REPL interface |

**Key takeaway:** Memory is what transforms a stateless LLM into a conversational agent. The modern approach uses `MessagesPlaceholder` in prompts and `RunnableWithMessageHistory` to manage history automatically. Choose your storage backend based on your deployment needs (see comparison below).

---

## Core Concepts at a Glance

| Concept | What It Does | Where to Start |
|---------|-------------|----------------|
| **Models** | Interface to LLM providers (Azure OpenAI, OpenAI, Anthropic) | Example 2 |
| **Messages** | Typed conversation turns (`SystemMessage`, `HumanMessage`, `AIMessage`) | Example 3 |
| **Prompt Templates** | Reusable, parameterized prompt structures | Example 5 |
| **LCEL** | Declarative chain composition with the `\|` pipe operator | Example 8 |
| **Output Parsers** | Extract and format model responses (e.g., `StrOutputParser`) | Example 8 |
| **Runnables** | Composable units of work (`RunnablePassthrough`, `RunnableWithMessageHistory`) | Example 12 |
| **Memory** | Conversation history management across turns and sessions | Example 15 |

---

## Navigating the Examples

**If you're brand new to LangChain:** Start at Example 1 and work through them in order. Each script is self-contained and can be run independently.

**If you know the basics and want to learn LCEL:** Jump to Example 8. Examples 8–14 form a complete LCEL mini-course.

**If you need to add memory to an existing app:** Start at Example 15 for the core pattern, then skip ahead to whichever storage backend matches your needs (17 for SQLite, 18 for Redis).

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
| `RedisChatMessageHistory` | High-traffic production apps | In-memory with optional persistence | Medium (requires Redis) |
| Window Memory (custom) | Long conversations on a budget | Depends on backing store | Low |
| Summary Memory (custom) | Very long conversations needing full context | Depends on backing store | Medium |

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
