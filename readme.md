# Machine learning in C

Training is a repository for my exploration journey of machine learning, starting from a simple perceptron to implementing neural networks.

## Usage

The code was not made for any specific use case, but if you want to experiment with it or copy parts, feel free to do so. `perceptron.c` implements a simple perceptron with no activation function, `gates.c` implements a neuron with two input bits being able to learn the logic gates `AND`, `OR` and `NAND`. However, this single cell struggles to understand the workings of an `XOR` gate, so I also implemented a minimal, fully connected, 2-layer neural network with such capabilities.

## License

The project is under the [MIT license](license.md). I must mention  [Tsoding's](www.github.com/tsoding/ml-notes) GitHub page, which sparked my interest in the topic. He makes awesome [youtube](www.youtube.com/tsoding-daily) videos about their coding projects, so make sure to check it out.