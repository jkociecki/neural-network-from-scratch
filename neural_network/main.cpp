#include "utils/MNISTLoader.h"
#include "nn/NeuralNetwork.h"
#include "utils/Serialization.h"
#include "optim/Adam.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>


int main() {
    // Path to MNIST dataset files
    std::string trainingImagesFile = "C:\\CLionProjects\\nn\\data\\t10k-images.idx3-ubyte";
    std::string trainingLabelsFile = "C:\\CLionProjects\\nn\\data\\t10k-labels.idx1-ubyte";
    std::string testImagesFile = "C:\\CLionProjects\\nn\\data\\train-images.idx3-ubyte";
    std::string testLabelsFile = "C:\\CLionProjects\\nn\\data\\train-labels.idx1-ubyte";

    // Load MNIST data
    MNISTLoader loader;
    std::vector<Matrix> allTrainImages = loader.loadImages(trainingImagesFile);
    std::vector<Matrix> allTrainLabels = loader.loadLabels(trainingLabelsFile, 10);
    std::vector<Matrix> testImages = loader.loadImages(testImagesFile);
    std::vector<Matrix> testLabels = loader.loadLabels(testLabelsFile, 10);

    std::vector<Matrix> trainImages, trainLabels, valImages, valLabels;
    loader.splitData(allTrainImages, allTrainLabels, trainImages, trainLabels, valImages, valLabels, 0.1);

    double learningRate = 0.001;
    auto adam = std::make_unique<Adam>(learningRate, 0.9, 0.999, 1e-8);
    NeuralNetwork network(0.0, std::move(adam));

    size_t inputSize = 784;
    size_t hiddenSize1 = 128;
    size_t hiddenSize2 = 64;
    size_t outputSize = 10;

    network.addLayer(inputSize, hiddenSize1, Activation::relu, Activation::reluDerivative, false);
    network.addLayer(hiddenSize1, hiddenSize2, Activation::relu, Activation::reluDerivative, false);
    network.addLayer(hiddenSize2, outputSize, Activation::identity, Activation::identityDerivative, true); // Softmax output

    std::cout << "Starting training...\n";

    int epochs = 5;
    int batchSize = 100;
    network.fit(trainImages, trainLabels, epochs, batchSize);

    double valAccuracy = network.evaluate(valImages, valLabels);
    std::cout << "Validation accuracy: " << valAccuracy * 100.0 << "%" << std::endl;

    double testAccuracy = network.evaluate(testImages, testLabels);
    std::cout << "Test accuracy: " << testAccuracy * 100.0 << "%" << std::endl;

    std::string modelFile = "mnist_model.bin";
    Serialization::serializeNetwork(network, modelFile);

    NeuralNetwork loadedNetwork = Serialization::deserializeNetwork(modelFile);

    double loadedAccuracy = loadedNetwork.evaluate(testImages, testLabels);
    std::cout << "Loaded model test accuracy: " << loadedAccuracy * 100.0 << "%" << std::endl;

    return 0;
}