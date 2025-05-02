//
// Created by Jędrzej on 2025-05-01.
//

#ifndef NN_SGD_H
#define NN_SGD_H
#include <unordered_map>
#include "optim/Optimizer.h"

class SGD : public Optimizer
{
public:
    explicit SGD(double lr = 0.01, double momentum = 0.0)
            : Optimizer(lr), momentum(momentum) {}

    virtual void update(Matrix& param, const Matrix& grad, const std::string& paramName) override;
    void reset() override;
    void serialize(std::ofstream& file) const override;
    std::string getType() const override;

private:
    double momentum;
    std::unordered_map<std::string, Matrix> velocities;

};


#endif //NN_SGD_H
