import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { ChatGroq } from '@langchain/groq';
import { StateGraph, MemorySaver, MessagesAnnotation } from '@langchain/langgraph';
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import { Annotation } from '@langchain/langgraph';
import readline from 'node:readline/promises';
import 'dotenv/config';

interface MessageState {
  messages: (HumanMessage | AIMessage)[];
}

// Add Memory
const checkpointer = new MemorySaver();

// Create the tool node
const tool = new TavilySearch({
  maxResults: 3,
  topic: "general",
  // includeAnswer: false,
  // includeRawContent: false,
  // includeImages: false,
  // includeImageDescriptions: false,
  // searchDepth: "basic",
  // timeRange: "day",
  // includeDomains: [],
  // excludeDomains: [],
});
const tools = [tool];
const toolNode = new ToolNode(tools);

// Get the llm from groq
const groq = new ChatGroq({
  model: 'llama-3.1-8b-instant',
  temperature: 0,
  maxRetries: 3,
  apiKey: process.env.GROQ_API_KEY,
}).bindTools(tools);//Binding the tools to the llm

// Function to decide whether to continue with llm or with tool
function shouldContinue(state: MessageState) {
  const lastMessage = state.messages[state.messages.length - 1] as any;

  // If the last message has tool calls, we need to execute them
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return 'tools';
  }
  // otherwise call the llm
  return '__end__';
}

// Agent model function
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

// Build the workflow using LangGraph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode('agent', agentModel)
  .addNode('tools', toolNode)//creating the tool node
  .addEdge('__start__', 'agent')
  .addConditionalEdges('agent', shouldContinue)//conditional edge to decide whether to go to tools or directly llm reponse
  .addEdge('tools', 'agent'); // After tools execute, go back to agent

const app = workflow.compile({
  checkpointer: checkpointer,
});

// Simple loading message functions
function showLoading(mode: 'start' | 'stop') {
  if (mode === 'start') {
    process.stdout.write('Loading...');
  } else {
    process.stdout.clearLine(0);
    process.stdout.cursorTo(0);
  }
}

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log('Chat bot started. Type "exit" to quit.');
  
  while (true) {
    const userInput = await rl.question('You: ');
    if (userInput.toLowerCase() === 'exit') {
      break;
    }

    if (!userInput.trim()) {
      continue; // Skip empty inputs
    }

    showLoading('start');
    
    try {
      const finalMessage = await app.invoke(
        { 
          messages: [new HumanMessage({content: userInput})] 
        },
        { 
          configurable: { 
            thread_id: '1' // Use unique thread_id for each conversation, for testing 1 is hardcoded
          } 
        }
      );
      
      showLoading('stop');
      
      // Handle case where messages might be empty
      if (finalMessage.messages && finalMessage.messages.length > 0) {
        const response = finalMessage.messages[finalMessage.messages.length - 1];
        console.log('Bot:', response?.content || 'No response content');
      } else {
        console.log('Bot: No response generated');
      }
      
    } catch (error) {
      showLoading('stop');
      console.error('Error:', error instanceof Error ? error.message : 'Unknown error');
    }
  }
  
  rl.close();
  console.log('Goodbye!');
}

// Handle process termination
process.on('SIGINT', () => {
  console.log('\nGoodbye!');
  process.exit(0);
});

main().catch(console.error);