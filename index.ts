import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { ChatGroq } from '@langchain/groq';
import { StateGraph,MessagesAnnotation} from '@langchain/langgraph';
import readline from 'node:readline/promises';
import 'dotenv/config';

interface MessageState {
    messages: string[];

}

//Get the llm from the groq
const groq = new ChatGroq({
    model: 'llama-3.1-8b-instant',
    temperature: 0,
    maxRetries: 3,
    apiKey: process.env.GROQ_API_KEY,

})


//Building a agent model and functionality
async function agentModel(state: MessageState){
    const response = await groq.invoke(state.messages);
    return { messages: [ new AIMessage(response)] }; // returning updated state with new message
}

//Build the workflow uisng LangGraph
const workflow = new StateGraph(MessagesAnnotation)
    .addNode('agent', agentModel)
    .addEdge('__start__', 'agent')
    .addEdge('agent','__end__');

 const app = workflow.compile();

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
        const finalMessage = await app.invoke({messages:[new HumanMessage(userInput)]});
        const response = finalMessage.messages[finalMessage.messages.length - 1];
        console.log('Bot:', response?.content);
    }
    rl.close();
    
}
main([]);
