
#ifndef NN_NEURALNETWORK_H
#define NN_NEURALNETWORK_H

#include "Layer.h"


class NeuralNetwork
{

public:
    NeuralNetwork(double learningRate = 0.01) : learningRate(learningRate) {}
    Matrix forward(const Matrix& input);
    double crossEntropyLoss(const Matrix& predicted, const Matrix& target);
    Matrix crossEntropyDerivative(const Matrix& predicted, const Matrix& target);
    void backward(const Matrix& target);
    void train(const Matrix& input, const Matrix& target);
    void fit(const std::vector<Matrix>& X, const std::vector<Matrix>& y, int epochs, int bachSize = 1);
    int predict(const Matrix& input);
    double evaluate(const std::vector<Matrix>& X, std::vector<Matrix>& y);


private:
    std::vector<Layer> layers;
    double learningRate;

};


#endif //NN_NEURALNETWORK_H
