# LangGraph Tool Calling Chatbot

A simple **tool-calling chatbot** built with **LangGraph**, **LangChain**, **Groq**, and **Tavily Search**.
This project demonstrates how an LLM can decide when to call an external tool, how LangGraph routes that tool call, and how the final response is generated after the tool result comes back.

---

## Overview

This project is a **basic agentic chatbot** that uses a **LangGraph workflow** to decide whether a user query needs an external tool.
If the LLM thinks a tool is required, it returns a **tool call**, LangGraph executes that tool through a **ToolNode**, and then the result is sent back to the LLM so it can generate the final answer.

In this project, the external tool is **Tavily Search**, which is used for web search.

So the chatbot can handle two types of flows:

* **Direct answer flow** → if the LLM can answer without a tool
* **Tool calling flow** → if the LLM decides that a web search is needed

---

# Tech Stack

* **TypeScript**
* **LangChain**
* **LangGraph**
* **Groq**
* **Tavily Search API**
* **Node.js**
* **dotenv**

---

# Features

* Chatbot built using **LangGraph StateGraph**
* **Tool calling support** using `bindTools()`
* **Tavily Search integration** for web search
* **Conditional routing** based on whether the LLM requested a tool
* **MemorySaver checkpointing** for thread-based state persistence
* Simple **CLI chat interface** using Node `readline`

---

# How It Works

The chatbot follows this high-level flow:

```text
User message
   ↓
Agent (LLM)
   ↓
Does the response contain a tool call?
   ├─ No  → End and return normal answer
   └─ Yes → Execute tool using ToolNode
               ↓
            Tool result
               ↓
            Agent (LLM again)
               ↓
         Final response to user
```

In LangGraph terms, the flow is:

```text
__start__ → agent → (tools or end)
tools → agent
```

---

# Project Goal

The purpose of this project is to learn and demonstrate:

* how **tool calling** works in LangChain / LangGraph
* how an LLM can decide when to use a tool
* how to route execution through a **graph**
* how to pass tool results back to the model
* how memory/checkpointing can be attached to a graph

This is a **tool-calling demo project**, not a full RAG system.

---

# Project Structure

```bash
.
├── src/
│   └── index.ts
├── .env
├── package.json
└── README.md
```

> If your file structure is different, update this section accordingly.

---

# Installation

## 1) Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

## 2) Install dependencies

```bash
npm install
```

## 3) Create a `.env` file

Add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# Run the Project

If you are using `ts-node`:

```bash
npx ts-node src/index.ts
```

Or if you have a script in `package.json`:

```bash
npm run dev
```

Once the app starts, you can chat in the terminal.

Example:

```bash
Chat bot started. Type "exit" to quit.
You: what is Node.js?
Bot: ...
```

---

# Core Concepts Used in This Project

This project is built around **5 main parts**:

1. **Messages** → stores user and AI chat messages
2. **LLM Agent** → decides whether to answer directly or call a tool
3. **ToolNode** → executes tools requested by the model
4. **Conditional routing** → decides whether the graph should go to the tool node
5. **MemorySaver** → stores graph state for a thread

---

# Code Walkthrough

---

# 1) Importing the required modules

```ts
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { ChatGroq } from '@langchain/groq';
import { StateGraph, MemorySaver, MessagesAnnotation } from '@langchain/langgraph';
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import readline from 'node:readline/promises';
import 'dotenv/config';
```

### What these imports do

* `HumanMessage` → represents a user message
* `AIMessage` → represents a model response
* `ChatGroq` → Groq LLM integration
* `StateGraph` → builds the graph workflow
* `MemorySaver` → stores graph checkpoints / state
* `MessagesAnnotation` → message-based graph state
* `ToolNode` → executes tools requested by the LLM
* `TavilySearch` → web search tool
* `readline` → terminal chat input

---

# 2) Message state

```ts
interface MessageState {
  messages: (HumanMessage | AIMessage)[];
}
```

This defines the state shape used in the project.

The chatbot stores conversation messages in a `messages` array.

Although the graph is later built using `MessagesAnnotation`, this interface helps conceptually explain that the state is message-based.

---

# 3) Adding memory / checkpointing

```ts
const checkpointer = new MemorySaver();
```

`MemorySaver` is used as the graph checkpointer.

## What it does

It stores the graph state for a given `thread_id`, so the conversation can continue across multiple messages in the same thread.

This means if the same `thread_id` is used for multiple invocations, the graph can remember previous messages for that conversation.

> Note: `MemorySaver` is not Redis or a database.
> It is an in-memory LangGraph checkpointing mechanism for storing graph state.

---

# 4) Creating the search tool

```ts
const tool = new TavilySearch({
  maxResults: 3,
  topic: "general",
});
```

This creates a **Tavily Search tool**.

## Purpose of this tool

The tool is used when the LLM decides that a user query needs web search.

Examples:

* recent events
* factual lookup
* information the model wants to verify
* questions that need external search

---

# 5) Creating the ToolNode

```ts
const tools = [tool];
const toolNode = new ToolNode(tools);
```

The `ToolNode` is responsible for **executing tools**.

This is important because the LLM only **requests** a tool call.
The LLM does **not** run the tool itself.

The actual flow is:

1. LLM returns a tool call
2. LangGraph routes execution to `ToolNode`
3. `ToolNode` runs the tool
4. Tool result is added back into the conversation
5. LLM uses that result to generate the final answer

---

# 6) Creating the Groq model and binding tools

```ts
const groq = new ChatGroq({
  model: 'llama-3.1-8b-instant',
  temperature: 0,
  maxRetries: 3,
  apiKey: process.env.GROQ_API_KEY,
}).bindTools(tools);
```

This is one of the most important parts of the project.

## Why `bindTools()` matters

```ts
.bindTools(tools)
```

This tells the LLM:

> “These are the tools you are allowed to call.”

Without `bindTools()`, the model can only generate plain text responses.
With `bindTools()`, it can return structured **tool calls**.

So if the model thinks a search is needed, it can output something like:

* tool name
* arguments for the tool
* query to search

LangGraph then reads that tool call and routes execution to the tool node.

---

# 7) The function that decides whether a tool should run

```ts
function shouldContinue(state: MessageState) {
  const lastMessage = state.messages[state.messages.length - 1] as any;

  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return 'tools';
  }

  return '__end__';
}
```

This is the **decision point** in the graph.

## What it checks

It looks at the **last AI message**.

If the last AI message contains a tool call:

```ts
lastMessage.tool_calls && lastMessage.tool_calls.length > 0
```

then the graph routes to the `tools` node.

Otherwise, the graph ends.

---

# This is the exact part of the code that decides whether a tool should be called

### If tool call exists:

```ts
return 'tools';
```

### If tool call does not exist:

```ts
return '__end__';
```

So in simple words:

* **Tool requested by model?** → go to `tools`
* **No tool requested?** → stop and return answer

---

# 8) The agent node

```ts
async function agentModel(state: MessageState) {
  try {
    const response = await groq.invoke(state.messages);
    return { messages: [response] };
  } catch (err) {
    console.error('Error in agentModel:', err);
    return { 
      messages: [new AIMessage("Sorry, I encountered an error processing your request.")] 
    };
  }
}
```

This is the **agent node** in the graph.

## What it does

It sends the current conversation messages to the Groq LLM:

```ts
const response = await groq.invoke(state.messages);
```

Then it returns the model’s response back into the graph state.

---

# Important behavior of the agent node

The same `agentModel()` function is used **before and after tool execution**.

## First time the agent runs

It sees the user message and decides:

* answer directly, or
* request a tool

## Second time the agent runs

After the tool result comes back, it sees:

* the user message
* its own tool call
* the tool result

Now it uses all of that to generate the final answer.

So the same agent node handles both:

* **tool decision**
* **final response generation**

---

# 9) Building the graph workflow

```ts
const workflow = new StateGraph(MessagesAnnotation)
  .addNode('agent', agentModel)
  .addNode('tools', toolNode)
  .addEdge('__start__', 'agent')
  .addConditionalEdges('agent', shouldContinue)
  .addEdge('tools', 'agent');
```

This is the heart of the project.

---

# Graph Explanation

## Add the `agent` node

```ts
.addNode('agent', agentModel)
```

This is the LLM node.

---

## Add the `tools` node

```ts
.addNode('tools', toolNode)
```

This node executes the tool calls.

---

## Start the graph from the agent

```ts
.addEdge('__start__', 'agent')
```

Whenever a user sends a message, the graph begins by sending it to the LLM.

---

## Add conditional routing after the agent

```ts
.addConditionalEdges('agent', shouldContinue)
```

After the agent runs, the graph asks:

* Did the model request a tool?
* If yes → go to `tools`
* If no → end

---

## Return from tools back to the agent

```ts
.addEdge('tools', 'agent');
```

Once the tool finishes, the result is passed back to the LLM so it can write the final answer.

---

# Final Graph Flow

## If no tool is needed

```text
User → agent → end
```

## If tool is needed

```text
User → agent → tools → agent → end
```

---

# 10) Compiling the graph

```ts
const app = workflow.compile({
  checkpointer: checkpointer,
});
```

This converts the graph definition into a runnable application.

The `checkpointer` is attached here so LangGraph can save state for each thread.

---

# 11) CLI chat interface

```ts
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
```

This project uses a simple terminal-based chat interface.

The user enters a message, the graph processes it, and the chatbot prints the final response.

---

# 12) Invoking the graph

```ts
const finalMessage = await app.invoke(
  { 
    messages: [new HumanMessage({content: userInput})] 
  },
  { 
    configurable: { 
      thread_id: '1'
    } 
  }
);
```

This is where the chatbot actually runs.

## What is passed into the graph

### State input

```ts
messages: [new HumanMessage({ content: userInput })]
```

This adds the latest user message into the graph.

### Configurable thread id

```ts
thread_id: '1'
```

This identifies the conversation thread for the checkpointer.

If the same thread id is reused, the graph can continue the same conversation state.

---

# 13) Getting the final response

```ts
if (finalMessage.messages && finalMessage.messages.length > 0) {
  const response = finalMessage.messages[finalMessage.messages.length - 1];
  console.log('Bot:', response?.content || 'No response content');
}
```

The graph may produce multiple messages during one run, such as:

1. Human message
2. AI message containing a tool call
3. Tool result message
4. Final AI response

That is why the code reads the **last message** from `finalMessage.messages`.
The last one is expected to be the final chatbot reply.

---

# Tool Calling Lifecycle in This Project

Here is the full tool-calling lifecycle from start to finish.

---

## Step 1: User asks a question

Example:

```text
You: what happened in Portugal match yesterday?
```

---

## Step 2: The graph sends the message to the agent

The `agentModel()` function runs:

```ts
const response = await groq.invoke(state.messages);
```

---

## Step 3: The model decides whether a tool is needed

If the model thinks it needs current web information, it returns a tool call.

That tool call is attached to the AI message.

---

## Step 4: `shouldContinue()` checks the last AI message

```ts
if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
  return 'tools';
}
```

If a tool call exists, the graph routes to the `tools` node.

---

## Step 5: `ToolNode` executes Tavily Search

The `ToolNode` runs the requested tool using the arguments returned by the model.

---

## Step 6: Tool result is added back into the conversation state

Now the graph contains:

* user question
* AI tool request
* tool output

---

## Step 7: Graph returns to the agent

```ts
.addEdge('tools', 'agent');
```

The LLM now sees the tool result and uses it to write the final answer.

---

## Step 8: Final answer is returned to the user

The last message in the graph state becomes the chatbot’s final response.

---

# Example Flow

Let’s say the user asks:

```text
You: who won the Portugal match yesterday?
```

## Possible execution flow

### 1. User message enters graph

`HumanMessage("who won the Portugal match yesterday?")`

### 2. Agent sees the message

The model decides it needs fresh information.

### 3. Agent returns a tool call

It asks to use Tavily Search.

### 4. `shouldContinue()` detects the tool call

Routes graph to `tools`.

### 5. `ToolNode` executes Tavily

Search query runs and results are returned.

### 6. Tool result goes back to the agent

The agent now has external information.

### 7. Agent writes the final answer

The final message is printed to the terminal.

---

# Why This Project Is Useful

This project is a clean learning example for understanding **tool calling in LangGraph**.

It teaches the exact pattern behind many modern AI agents:

* let the LLM decide whether a tool is needed
* execute the tool in a controlled node
* return the result back to the LLM
* let the LLM write the final response

This same pattern can later be extended to:

* RAG document retrieval
* database lookup
* API calling
* calculator tools
* email tools
* weather tools
* multi-tool agent systems

---

# Difference Between This Project and a Full RAG System

This project is **not a full Retrieval-Augmented Generation (RAG) system**.

## In this project

* the external tool is **Tavily Search**
* the tool retrieves information from the web

## In a RAG system

* the external retriever usually searches a **vector database**
* retrieved chunks are injected into the prompt
* the model answers based on those chunks

So this project is best understood as a **tool-calling agent example**, not a complete RAG architecture.

---

# Current Limitations

This project is intentionally simple and mainly for learning.

Some limitations:

* only one tool is used (`TavilySearch`)
* no custom system prompt for stronger tool usage instructions
* `thread_id` is hardcoded as `'1'`
* no persistent database storage
* no custom RAG retrieval layer
* no frontend interface
* no user/session management

---

# Possible Improvements

If this project is extended further, the following improvements can be added:

## 1) Use a dynamic `thread_id`

Instead of hardcoding:

```ts
thread_id: '1'
```

use a unique session or conversation id.

---

## 2) Add a system prompt

A system prompt can guide the model on **when to use tools** and **when to answer directly**.

---

## 3) Add more tools

Examples:

* calculator tool
* weather tool
* custom database lookup tool
* vector search / RAG tool
* document retrieval tool

---

## 4) Replace Tavily-only flow with multi-tool routing

The model could choose between:

* direct answer
* knowledge base retrieval
* web search
* custom APIs

---

## 5) Add proper production memory

Instead of only using `MemorySaver`, production systems often use:

* Redis for short-term memory
* database storage for long-term history
* vector DB for knowledge retrieval

---

# Example Use Cases

This project is useful for learning and experimenting with:

* **web-search assistants**
* **tool-calling chatbots**
* **LangGraph agent workflows**
* **LLM orchestration**
* **agentic RAG foundations**
* **multi-step AI pipelines**

---

# Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# Main Learning Takeaway

This project demonstrates the core tool-calling loop:

```text
User → LLM → Tool Decision → Tool Execution → LLM → Final Response
```

The most important concept is that the **LLM does not directly execute the tool**.
Instead:

1. it **requests** the tool,
2. LangGraph **routes** the request,
3. `ToolNode` **executes** it,
4. and the LLM uses the tool result to produce the final answer.

---

# Author

**Sudip Sharma**

If you’re exploring **LangGraph**, **LangChain**, **tool calling**, or building toward an **agentic RAG system**, this project is a great starting point to understand how the workflow fits together.

---
