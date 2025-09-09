import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { ChatGroq } from '@langchain/groq';
import { StateGraph,MessagesAnnotation} from '@langchain/langgraph';
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import readline from 'node:readline/promises';
import 'dotenv/config';

interface MessageState {
    messages: BaseMessage[];
}

//Create the tool node
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
const tools= [tool];//can add multiple tools
const toolNode = new ToolNode(tools);

//Get the llm from the groq
const groq = new ChatGroq({
    model: 'llama-3.1-8b-instant',
    temperature: 0,
    maxRetries: 3,
    apiKey: process.env.GROQ_API_KEY,
}).bindTools(tools);//Binding the tools to the llm


//Function to decide whether to continue with llm or with tool
function shouldContinue({ messages }: typeof MessagesAnnotation.State){
    const lastMessage = messages[messages.length - 1] as  any;
    // console.log('Last message:', lastMessage);
    if (lastMessage instanceof AIMessage && lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
        // console.log("Calling the tools");
        return 'tools';
    }
    // console.log("Calling the llm");
    return '__end__';
}


//Building a agent model and functionality
async function agentModel(state: MessageState){
    try{
        const response = await groq.invoke(state.messages);
        return { messages: [response] }; // returning updated state with new message
    }catch(err){
         return { 
      messages: [new AIMessage("Sorry, I encountered an error processing your request.")] 
    };
    }
}

//Build the workflow uisng LangGraph
const workflow = new StateGraph(MessagesAnnotation)
    .addNode('agent', agentModel)
    .addEdge('__start__', 'agent')
    .addNode('tools', toolNode)//creating the tool node
    .addEdge('tools','agent')
    .addConditionalEdges('agent', shouldContinue)
    .addEdge('agent','__end__');

 const app = workflow.compile();

 // Simple loading message functions
function showLoading(mode: 'start' | 'stop') {
  if (mode === 'start') {
    process.stdout.write('Loading...');
  } else {
    process.stdout.clearLine(0);
    process.stdout.cursorTo(0);
  }
}

async function main(params: string[]) {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    while(true){
        const userInput = await rl.question('You:');
        if(userInput.toLowerCase() === 'exit'){
            break;
        }
        showLoading('start');
        const finalMessage = await app.invoke({messages:[new HumanMessage(userInput)]});
        const response = finalMessage.messages[finalMessage.messages.length - 1];
        showLoading('stop');
        console.log('Bot:', response?.content);
    }
    rl.close();
    
}
main([]);
