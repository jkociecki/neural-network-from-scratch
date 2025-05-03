
#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include <cmath>
#include "Matrix.h"

namespace Activation
{
    inline double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double sigmoid_derivative(double x)
    {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    inline double relu(double x)
    {
        return std::max(0.0, x);
    }

    inline double reluDerivative(double x)
    {
        return x > 0 ? 1.0 : 0.0;
    }

    inline double identity(double x)
    {
        return x;
    }

    inline double identityDerivative(double x)
    {
        return 1.0;
    }

    inline Matrix softmax(const Matrix& x) {
        double max_val = x.at(0, 0);
        for (size_t i = 0; i < x.getRows(); ++i) {
            for (size_t j = 0; j < x.getCols(); ++j) {
                if (x.at(i, j) > max_val) max_val = x.at(i, j);
            }
        }

        Matrix exp_values = x.apply([max_val](double val) {
            return std::exp(val - max_val);
        });

        double sum = 0.0;
        for (size_t i = 0; i < exp_values.getRows(); ++i) {
            for (size_t j = 0; j < exp_values.getCols(); ++j) {
                sum += exp_values.at(i, j);
            }
        }

        return exp_values.apply([sum](double val) { return val / sum; });
    }


}


#endif //NN_ACTIVATION_H
