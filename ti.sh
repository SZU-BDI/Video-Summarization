node -e "(async(u)=>{console.log(await require('terminal-image').file(process.argv[1]))})()" $*
