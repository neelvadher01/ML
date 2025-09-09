const readline = require("readline");
const { Readline } = require("readline/promises");
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
//////////////////////////////////////////
rl.question("Enter a number: ", function(num){
    let a=parseInt(num);
    const N=a;
    let binary="";
    while(a>0){
    binary=(a%2)+binary;
    a=Math.floor(a/2);
    }
    console.log("Binary of "+N+" is: "+binary);
///////////////////////////////////////////
rl.close();
});


