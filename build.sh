#!/bin/zsh

set -xe

mkdir -p build/
gcc -Wall -Wextra -o build/perceptron perceptron.c -lm
gcc -Wall -Wextra -o build/gates gates.c -lm
gcc -Wall -Wextra -o build/neural neural.c -lm
