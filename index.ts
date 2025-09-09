import { StateGraph,MessagesAnnotation} from '@langchain/langgraph';
import readline from 'node:readline/promises';

interface MessageState {
    messages: string[];

}


//Building a agent model and functionality
function agentModel(state: MessageState){
    console.log("calling agent");
    return state;
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
        const finalMessage = await app.invoke({messages:[userInput]});
        const response = finalMessage.messages[finalMessage.messages.length - 1];
        console.log('Bot:', response?.content);
    }
    rl.close();
    
}
main([]);
