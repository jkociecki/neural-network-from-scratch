#ifndef NN_LAYER_H
#define NN_LAYER_H
#include <utility>
#include "optim/Optimizer.h"
#include "Matrix.h"
#include "Activation.h"

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
            isSoftmaxLayer(isSoftmax),
            name("layer_" + std::to_string(layerCounter++)) {}

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& error, double learningRate);
    Matrix backward(const Matrix& error, double learningRate, Optimizer& optimizer);

    [[nodiscard]] const Matrix& getOutput() const { return output; }
    [[nodiscard]] Matrix& getWeights() { return weights; }
    [[nodiscard]] Matrix& getBiases() { return biases; }
    [[nodiscard]] const Matrix& getWeights() const { return weights; }
    [[nodiscard]] const Matrix& getBiases() const { return biases; }
    [[nodiscard]] bool getIsSoftmax() const { return isSoftmaxLayer; }
    [[nodiscard]] const Matrix& getPreActivation() const { return preActivation; }
    [[nodiscard]] std::function<double(double)> getActivationDerivative() const { return activationDerivative; }
    [[nodiscard]] const std::string& getName() const { return name; }

    void setWeights(const Matrix& w) { weights = w; }
    void setBiases(const Matrix& b) { biases = b; }

private:
    Matrix weights;
    Matrix biases;
    Matrix input;
    Matrix output;
    Matrix preActivation;

    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;

    bool isSoftmaxLayer;
    std::string name;
    static int layerCounter;


};

inline int Layer::layerCounter = 0;


#endif //NN_LAYER_H