//
// Created by Jędrzej on 2025-05-01.
//

#include "optim/SGD.h"


void SGD::update(Matrix &param, const Matrix &grad, const std::string &paramName)
{
    if (momentum > 0.0)
    {
        if (velocities.find(paramName) == velocities.end())
        {
            velocities[paramName] = Matrix::zeros(param.getRows(), param.getCols());
        }
        velocities[paramName] = velocities[paramName] * momentum - grad * learningRate;
        param = param + velocities[paramName];
    }
    else
    {
        param = param - grad * learningRate;
    }
}

void SGD::reset()
{
    velocities.clear();
}

void SGD::serialize(std::ofstream &file) const
{
    std::string type = getType();
    size_t typeLen = type.length();
    file.write(reinterpret_cast<const char*>(&typeLen), sizeof(typeLen));
    file.write(type.c_str(), typeLen);

    file.write(reinterpret_cast<const char*>(&learningRate), sizeof(learningRate));
    file.write(reinterpret_cast<const char*>(&momentum), sizeof(momentum));
}

std::string SGD::getType() const
{
    return "SGD";
}


