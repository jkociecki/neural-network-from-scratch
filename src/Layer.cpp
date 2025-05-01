
#include "../include/Layer.h"

Matrix Layer::forward(const Matrix &input)
{
    this->input = input;
    preActivation = weights.dot(input) + biases;

    if (isSoftmaxLayer)
        output = Activation::softmax(preActivation);
    else
        output = preActivation.apply(activation);

}

Matrix Layer::backward(const Matrix &error, double learningRate)
{

    Matrix gradient(error.getRows(), error.getCols());

    if (isSoftmaxLayer)
        gradient = error;
    else
        gradient = preActivation.apply(activationDerivative) * error;

    Matrix prevLayerError = weights.transpose().dot(gradient);
    Matrix weightsDelta = gradient.dot(input.transpose()) * learningRate;
    Matrix biasesDelta = gradient * learningRate;

    weights = weights - weightsDelta;
    biases = biases - biasesDelta;

    return prevLayerError;

}
