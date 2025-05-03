#include <iostream>
#include <string>
#include <sstream>
#include "include/nn/NeuralNetwork.h"
#include "utils/Serialization.h"

Matrix preprocessInput(const std::vector<double>& rawData)
{
    Matrix input(rawData.size(), 1);

    for (size_t i = 0; i < rawData.size(); i++) {
        input.at(i, 0) = rawData[i] /255.0;
    }

    return input;
}

int main()
{

    std::string line;
    std::getline(std::cin, line);
    std::istringstream iss(line);
    std::vector<double> inputData;
    double value;

    while (iss >> value)
    {
        inputData.push_back(value);
    }

    Matrix input = preprocessInput(inputData);

    std::string modelFile = "C:\\CLionProjects\\nn\\neural_network\\mnist_model.bin";
    NeuralNetwork model = Serialization::deserializeNetwork(modelFile);

    int prediction = model.predict(input);
    std::cout << prediction << std::endl;

    return 0;

}