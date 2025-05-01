
#include "../include/NeuralNetwork.h"

Matrix NeuralNetwork::forward(const Matrix &input)
{
    Matrix currentOutput = input;

    for (Layer& layer : layers)
    {
        currentOutput = layer.forward((currentOutput));
    }
    return currentOutput;
}

double NeuralNetwork::crossEntropyLoss(const Matrix &predicted, const Matrix &target)
{
    double loss = 0.0;
    for (size_t i = 0; i < predicted.getRows(); i++)
    {
        for (size_t j = 0; j < predicted.getCols(); j++)
        {
            double p = std::max(predicted.at(i, j), 1e-15);
            loss -= target.at(i, j) * std::log(p);
        }
    }
}

Matrix NeuralNetwork::crossEntropyDerivative(const Matrix &predicted, const Matrix &target)
{
    return predicted - target;
}

void NeuralNetwork::backward(const Matrix &target)
{
    Matrix output = layers.back().getOutput();
    Matrix error = crossEntropyDerivative(output, target);

    for (auto it = layers.rbegin(); it != layers.rend(); it++)
    {
        error = it->backward(error, learningRate);
    }
}

void NeuralNetwork::train(const Matrix &input, const Matrix &target)
{
    forward(input);
    backward(target);
}

void NeuralNetwork::fit(const std::vector<Matrix> &X, const std::vector<Matrix> &y, int epochs, int bachSize) {
    size_t numSamples = X.size();

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < numSamples; i++) {
            Matrix predicted = forward((X[i]));

            totalLoss += crossEntropyLoss(predicted, y[i]);

            backward(y[i]);

            if ((i + 1) % 100 == 0 || i == numSamples - 1) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
                          << ", Sample " << i + 1 << "/" << numSamples
                          << ", Loss: " << totalLoss / (i + 1) << std::endl;
            }
        }

    }
}

int NeuralNetwork::predict(const Matrix &input)
{
    Matrix output = forward(input);
    int predictedClass = 0;
    double maxValue = output.at(0, 0);

    for (size_t i = 1; i < output.getRows(); ++i) {
        if (output.at(i, 0) > maxValue) {
            maxValue = output.at(i, 0);
            predictedClass = i;
        }
    }

    return predictedClass;
}

double NeuralNetwork::evaluate(const std::vector<Matrix> &X, std::vector<Matrix> &y)
{
    size_t numSamples = X.size();
    size_t correct = 0;

    for (size_t i = 0; i < numSamples; ++i) {
        int predicted = predict(X[i]);

        int trueClass = 0;
        for (size_t j = 1; j < y[i].getRows(); ++j) {
            if (y[i].at(j, 0) > y[i].at(trueClass, 0)) {
                trueClass = j;
            }
        }

        if (predicted == trueClass) {
            correct++;
        }
    }

    return static_cast<double>(correct) / numSamples;
}

void NeuralNetwork::addLayer(size_t inputSize, size_t outputSize,
                             const std::function<double(double)>& activation,
                             const std::function<double(double)>& activationDerivative,
                             bool isSoftmax)
{
    layers.emplace_back(inputSize, outputSize, activation, activationDerivative, isSoftmax);
}
