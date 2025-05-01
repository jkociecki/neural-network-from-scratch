
#include "../include/NeuralNetwork.h"

Matrix NeuralNetwork::forward(const Matrix &input) {
    return Matrix(0, 0);
}

double NeuralNetwork::crossEntropyLoss(const Matrix &predicted, const Matrix &target) {
    return 0;
}

Matrix NeuralNetwork::crossEntropyDerivative(const Matrix &predicted, const Matrix &target) {
    return Matrix(0, 0);
}

void NeuralNetwork::backward(const Matrix &target) {

}

void NeuralNetwork::train(const Matrix &input, const Matrix &target) {

}

void NeuralNetwork::fit(const std::vector<Matrix> &X, const std::vector<Matrix> &y, int epochs, int bachSize) {

}

int NeuralNetwork::predict(const Matrix &input) {
    return 0;
}

double NeuralNetwork::evaluate(const std::vector<Matrix> &X, std::vector<Matrix> &y) {
    return 0;
}
