//
// Created by Jędrzej on 2025-05-01.
//

#ifndef NN_ADAM_H
#define NN_ADAM_H

#include <unordered_map>
#include "optim/Optimizer.h"

class Adam : public Optimizer
{
public:
    explicit Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
            : Optimizer(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void update(Matrix& param, const Matrix& grad, const std::string& paramName) override;

    void reset() override;
    void serialize(std::ofstream& file) const override;
    [[nodiscard]]  std::string getType() const override;


private:
    double beta1;
    double beta2;
    double epsilon;
    unsigned long long t;
    std::unordered_map<std::string, Matrix> m;
    std::unordered_map<std::string, Matrix> v;

};


#endif //NN_ADAM_H
