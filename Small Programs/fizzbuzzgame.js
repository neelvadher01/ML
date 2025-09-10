// Here is how the game works: We count from 1 to a given number ("1, 2, 3, ..."):

// If the next number is a multiple of 3, we say "Fizz".
// If the next number is a multiple of 5, we say "Buzz".
// If the next number is a multiple of both 3 and 5, we say "FizzBuzz"!

let given_no=15
for(i=1;i<=given_no;i++){
    if(i%3!==0 & i%5!==0){
        console.log(i);
    }
    else if(i%3==0 & i%5==0){
        console.log("FizzBuzz");
    }
    else if(i%3==0){
        console.log("Fizz");
    }
    else{
        console.log("Buzz");
    }
}