//
// Created by Jędrzej on 2025-05-01.
//

#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "nn/NeuralNetwork.h"
#include <string>
#include <fstream>

namespace Serialization
{

    void serializeMatrix(std::ofstream& file, const Matrix& matrix);
    Matrix deserializeMatrix(std::ifstream& file);

    void serializeNetwork(const NeuralNetwork& network, const std::string& filename);
    NeuralNetwork deserializeNetwork(const std::string& filename);

}

#endif // SERIALIZATION_H

