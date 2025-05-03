//
// Created by Jędrzej on 2025-05-01.
//

#include "utils/Serialization.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace Serialization
{

    void serializeMatrix(std::ofstream& file, const Matrix& matrix)
    {
        size_t rows = matrix.getRows();
        size_t cols = matrix.getCols();

        file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<char*>(&cols), sizeof(cols));

        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                double value = matrix.at(i, j);
                file.write(reinterpret_cast<char*>(&value), sizeof(value));
            }
        }
    }

    Matrix deserializeMatrix(std::ifstream& file)
    {
        size_t rows, cols;

        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        Matrix matrix(rows, cols);

        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                double value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                matrix.at(i, j) = value;
            }
        }

        return matrix;
    }

    std::map<std::string, std::string> loadConfig(const std::string& filename) {
        std::map<std::string, std::string> config;
        std::ifstream file(filename);
        std::string line;

        while (std::getline(file, line)) {
            std::istringstream is_line(line);
            std::string key;
            if (std::getline(is_line, key, '=')) {
                std::string value;
                if (std::getline(is_line, value)) {
                    config[key] = value;
                }
            }
        }
        return config;
    }

    void serializeNetwork(const NeuralNetwork& network, const std::string& filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            std::cerr << "Cannot open file for serialization: " << filename << std::endl;
            return;
        }

        size_t numLayers = network.getLayers().size();
        file.write(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

        for (const auto& layer : network.getLayers())
        {
            serializeMatrix(file, layer.getWeights());
            serializeMatrix(file, layer.getBiases());

            bool isSoftmax = layer.getIsSoftmax();
            file.write(reinterpret_cast<char*>(&isSoftmax), sizeof(isSoftmax));
        }

        double learningRate = network.getLearningRate();
        file.write(reinterpret_cast<char*>(&learningRate), sizeof(learningRate));

        std::cout << "Model saved to: " << filename << std::endl;
    }

    NeuralNetwork deserializeNetwork(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open file for deserialization: " << filename << std::endl;
            return NeuralNetwork(0.01);
        }

        size_t numLayers;
        file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

        double learningRate;

        size_t currentPos = file.tellg();
        file.seekg(-(long)sizeof(double), std::ios::end);
        file.read(reinterpret_cast<char*>(&learningRate), sizeof(learningRate));
        file.seekg(currentPos);

        NeuralNetwork network(learningRate);

        for (size_t i = 0; i < numLayers; i++)
        {
            Matrix weights = deserializeMatrix(file);
            Matrix biases = deserializeMatrix(file);

            bool isSoftmax;
            file.read(reinterpret_cast<char*>(&isSoftmax), sizeof(isSoftmax));

            if (isSoftmax)
            {
                network.addLayerWithParams(weights, biases, Activation::identity, Activation::identityDerivative, true);
            }
            else
            {
                network.addLayerWithParams(weights, biases, Activation::relu, Activation::reluDerivative, false);
            }
        }

        return network;
    }

}
