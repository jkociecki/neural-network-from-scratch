//
// Created by Jędrzej on 2025-05-01.
//

#include <valarray>
#include "optim/Adam.h"


void Adam::update(Matrix &param, const Matrix &grad, const std::string &paramName)
{
    t++;

    if (m.find(paramName) == m.end())
    {
        m[paramName] = Matrix::zeros(param.getRows(), param.getCols());
        v[paramName] = Matrix::zeros(param.getRows(), param.getCols());
    }

    for (size_t i = 0; i < param.getRows(); i++)
    {
        for (size_t j = 0; j < param.getCols(); j++)
        {
            m[paramName].at(i, j) = beta1 * m[paramName].at(i, j) + (1.0 - beta1) * grad.at(i, j);
        }
    }

    for (size_t i = 0; i < param.getRows(); i++)
    {
        for (size_t j = 0; j < param.getCols(); j++)
        {
            double g = grad.at(i, j);
            v[paramName].at(i, j) = beta2 * v[paramName].at(i, j) + (1.0 - beta2) * (g * g);
        }
    }

    double correction1 = 1.0 - std::pow(beta1, t);
    double correction2 = 1.0 - std::pow(beta2, t);

    for (size_t i = 0; i < param.getRows(); i++)
    {
        for (size_t j = 0; j < param.getCols(); j++)
        {
            double m_corrected = m[paramName].at(i, j) / correction1;
            double v_corrected = v[paramName].at(i, j) / correction2;

            param.at(i, j) -= learningRate * m_corrected / (std::sqrt(v_corrected) + epsilon);
        }
    }
}

void Adam::reset()
{
    m.clear();
    v.clear();
    t = 0;
}

void Adam::serialize(std::ofstream &file) const
{
    std::string type = getType();
    size_t typeLen = type.length();
    file.write(reinterpret_cast<const char*>(&typeLen), sizeof(typeLen));
    file.write(type.c_str(), typeLen);

    file.write(reinterpret_cast<const char*>(&learningRate), sizeof(learningRate));
    file.write(reinterpret_cast<const char*>(&beta1), sizeof(beta1));
    file.write(reinterpret_cast<const char*>(&beta2), sizeof(beta2));
    file.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    file.write(reinterpret_cast<const char*>(&t), sizeof(t));
}

std::string Adam::getType() const
{
    return "Adam";
}