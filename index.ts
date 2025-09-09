import readline from 'node:readline/promises';



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
        console.log(`Hello, ${userInput}!`);
    }
    rl.close();
    
}
main([]);
