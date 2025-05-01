
#ifndef NN_LAYER_H
#define NN_LAYER_H
#include <utility>

#include "Matrix.h"


class Layer
{

public:
    Layer(size_t inputSize, size_t outputSize,
          std::function<double(double)> activation,
          std::function<double(double)> activationDerivative,
          bool isSoftmax = false,
          double weightInit = 0.01)
          :
          weights(Matrix::random(outputSize, inputSize, -weightInit, weightInit)),
          biases(Matrix(outputSize, 1, 0.0)),
          input(Matrix(inputSize, 1.0)),
          output(Matrix(outputSize, 1.0)),
          preActivation(Matrix(outputSize, 1.0)),
          activation(std::move(activation)),
          activationDerivative(std::move(activationDerivative)),
          isSoftmaxLayer(isSoftmax) {}

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& error, double learningRate);




private:
    Matrix weights;
    Matrix biases;
    Matrix input;
    Matrix output;
    Matrix preActivation;

    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;

    bool isSoftmaxLayer;


};


#endif //NN_LAYER_H
