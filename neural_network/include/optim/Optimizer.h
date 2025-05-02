//
// Created by Jędrzej on 2025-05-01.
//

#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H
#include "nn/Matrix.h"
#include "fstream"

class Optimizer
{
public:
    explicit Optimizer(double lr = 0.01) : learningRate(lr) {}

    virtual void update(Matrix& param, const Matrix& grad, const std::string& paramName) = 0;
    virtual void reset() = 0;
    virtual void serialize(std::ofstream& file) const = 0;
    [[nodiscard]] virtual std::string getType() const = 0;

    [[nodiscard]] double getLearningRate() const { return learningRate; }
    void setLearningRate(double lr) { learningRate = lr; }

protected:
    double learningRate = 0.0;
};


#endif //NN_OPTIMIZER_H
