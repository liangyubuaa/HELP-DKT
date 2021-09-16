#!/bin/bash
if [ -z "$1" ]; then
    echo "Required parameter c(correct programs)/a(all programs) missing!"
    exit 1
fi

echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "ASTs are generating..."
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+"
# AST_generation
cd AST_generation
if [ "$1" == "c" ]; then
    bigcode-astgen-py --batch -o asts "../../Data/Original_Codes/c_*.py"
elif [ "$1" == "a" ]; then
    bigcode-astgen-py --batch -o asts "../../Data/Original_Codes/*.py"
else
    echo "Wrong parameter! You should enter c or a!"
    exit 1
fi

cd ..
echo "+-+-+-+-+-+-+"
echo "success!"
echo "+-+-+-+-+-+-+"
