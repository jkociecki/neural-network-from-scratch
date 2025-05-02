
#ifndef NN_NEURALNETWORK_H
#define NN_NEURALNETWORK_H

#include "nn/Layer.h"
#include <memory>
#include <functional>

class NeuralNetwork
{

public:
    explicit NeuralNetwork(double learningRate = 0.01)
        : learningRate(learningRate), optimizer(nullptr) {}
    NeuralNetwork(double learningRate, std::unique_ptr<Optimizer> optimizer)
        : learningRate(learningRate), optimizer(std::move(optimizer)) {}

    Matrix forward(const Matrix& input);
    static double crossEntropyLoss(const Matrix& predicted, const Matrix& target);
    static Matrix crossEntropyDerivative(const Matrix& predicted, const Matrix& target);
    void backward(const Matrix& target);
    void train(const Matrix& input, const Matrix& target);
    void fit(const std::vector<Matrix>& X, const std::vector<Matrix>& y, int epochs, int bachSize = 1);
    int predict(const Matrix& input);
    double evaluate(const std::vector<Matrix>& X, std::vector<Matrix>& y);


    [[nodiscard]] const std::vector<Layer>& getLayers() const { return layers; }
    [[nodiscard]] double getLearningRate() const { return learningRate; }

    void addLayerWithParams(const Matrix& weights, const Matrix& biases,
                            const std::function<double(double)>& activation,
                            const std::function<double(double)>& activationDerivative,
                            bool isSoftmax = false);
    void addLayer(size_t inputSize, size_t outputSize,
                  const std::function<double(double)>& activation = Activation::sigmoid,
                  const std::function<double(double)>& activationDerivative = Activation::sigmoid_derivative,
                  bool isSoftmax = false);

private:
    std::vector<Layer> layers;
    std::unique_ptr<Optimizer> optimizer;
    double learningRate;


};


#endif //NN_NEURALNETWORK_H
