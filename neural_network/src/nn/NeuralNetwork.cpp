#include <random>

#include "nn/NeuralNetwork.h"

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

    return loss;
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
        if (optimizer)
            error = it->backward(error, learningRate, *optimizer);
        else
            error = it->backward(error, learningRate);
    }
}

void NeuralNetwork::train(const Matrix &input, const Matrix &target)
{
    forward(input);
    backward(target);
}

void NeuralNetwork::fit(const std::vector<Matrix> &X, const std::vector<Matrix> &y, int epochs, int batchSize)
{
    size_t numSamples = X.size();
    batchSize = std::min(batchSize, static_cast<int>(numSamples));
    size_t numBatches = (numSamples + batchSize - 1) / batchSize;

    std::vector<Matrix> accumulatedWeightGradients;
    std::vector<Matrix> accumulatedBiasGradients;

    for (const auto& layer : layers)
    {
        accumulatedWeightGradients.push_back(Matrix::zeros(layer.getWeights().getRows(), layer.getWeights().getCols()));
        accumulatedBiasGradients.push_back(Matrix::zeros(layer.getBiases().getRows(), layer.getBiases().getCols()));
    }

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double totalLoss = 0.0;

        std::vector<size_t> indices(numSamples);
        for (size_t i = 0; i < numSamples; i++)
        {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

        for (size_t batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            size_t startIdx = batchIdx * batchSize;
            size_t endIdx = std::min(startIdx + batchSize, numSamples);
            size_t currentBatchSize = endIdx - startIdx;

            for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++)
            {
                accumulatedWeightGradients[layerIdx] = Matrix::zeros(
                        layers[layerIdx].getWeights().getRows(),
                        layers[layerIdx].getWeights().getCols()
                );
                accumulatedBiasGradients[layerIdx] = Matrix::zeros(
                        layers[layerIdx].getBiases().getRows(),
                        layers[layerIdx].getBiases().getCols()
                );
            }

            double batchLoss = 0.0;

            for (size_t i = startIdx; i < endIdx; i++)
            {
                size_t sampleIdx = indices[i];

                Matrix predicted = forward(X[sampleIdx]);

                double loss = crossEntropyLoss(predicted, y[sampleIdx]);
                batchLoss += loss;

                Matrix output = layers.back().getOutput();
                Matrix error = crossEntropyDerivative(output, y[sampleIdx]);

                for (size_t layerIdx = layers.size(); layerIdx > 0; layerIdx--) {
                    Layer& layer = layers[layerIdx - 1];

                    Matrix gradient;
                    if (layer.getIsSoftmax()) {
                        gradient = error;
                    }
                    else
                    {
                        Matrix preActivation = layer.getPreActivation();
                        Matrix activationDeriv = preActivation.apply(layer.getActivationDerivative());

                        gradient = Matrix(error.getRows(), error.getCols());
                        for (size_t r = 0; r < gradient.getRows(); r++)
                        {
                            for (size_t c = 0; c < gradient.getCols(); c++)
                            {
                                gradient.at(r, c) = error.at(r, c) * activationDeriv.at(r, c);
                            }
                        }
                    }

                    if (layerIdx > 1)
                    {
                        error = layer.getWeights().transpose().dot(gradient);
                    }

                    Matrix input = (layerIdx > 1) ? layers[layerIdx - 2].getOutput() : X[sampleIdx];
                    Matrix weightGradient = gradient.dot(input.transpose());
                    accumulatedWeightGradients[layerIdx - 1] = accumulatedWeightGradients[layerIdx - 1] + weightGradient;
                    accumulatedBiasGradients[layerIdx - 1] = accumulatedBiasGradients[layerIdx - 1] + gradient;
                }
            }

            for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++)
            {
                accumulatedWeightGradients[layerIdx] = accumulatedWeightGradients[layerIdx] * (1.0 / currentBatchSize);
                accumulatedBiasGradients[layerIdx] = accumulatedBiasGradients[layerIdx] * (1.0 / currentBatchSize);

                if (optimizer)
                {
                    optimizer->update(layers[layerIdx].getWeights(), accumulatedWeightGradients[layerIdx],
                                      "layer_" + std::to_string(layerIdx) + "_weights");
                    optimizer->update(layers[layerIdx].getBiases(), accumulatedBiasGradients[layerIdx],
                                      "layer_" + std::to_string(layerIdx) + "_biases");
                }
                else
                {
                    layers[layerIdx].setWeights(layers[layerIdx].getWeights() -
                                                accumulatedWeightGradients[layerIdx] * learningRate);
                    layers[layerIdx].setBiases(layers[layerIdx].getBiases() -
                                               accumulatedBiasGradients[layerIdx] * learningRate);
                }
            }

            totalLoss += batchLoss;

            if ((batchIdx + 1) % 10 == 0 || batchIdx == numBatches - 1)
            {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
                          << ", Batch " << batchIdx + 1 << "/" << numBatches
                          << ", Average Loss: " << totalLoss / ((batchIdx + 1) * currentBatchSize) << std::endl;
            }
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " completed, Average Loss: " << totalLoss / numSamples << std::endl;
    }
}

int NeuralNetwork::predict(const Matrix &input)
{
    Matrix output = forward(input);
    int predictedClass = 0;
    double maxValue = output.at(0, 0);

    for (size_t i = 1; i < output.getRows(); ++i)
    {
        if (output.at(i, 0) > maxValue)
        {
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

    for (size_t i = 0; i < numSamples; ++i)
    {
        int predicted = predict(X[i]);

        size_t trueClass = 0;
        for (size_t j = 1; j < y[i].getRows(); ++j)
        {
            if (y[i].at(j, 0) > y[i].at(trueClass, 0))
            {
                trueClass = j;
            }
        }

        if (predicted == trueClass)
        {
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

void NeuralNetwork::addLayerWithParams(const Matrix &weights, const Matrix &biases,
                                       const std::function<double(double)> &activation,
                                       const std::function<double(double)> &activationDerivative, bool isSoftmax)
{
    Layer layer(weights.getRows(), weights.getCols(), activation, activationDerivative, isSoftmax);

    layer.setWeights(weights);
    layer.setBiases(biases);

    layers.push_back(layer);
}