#include "utils/MNISTLoader.h"
#include "nn/NeuralNetwork.h"
#include "utils/Serialization.h"
#include "optim/Adam.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>


int main() {

    auto config = Serialization::loadConfig("config.txt");
    std::string trainingImagesFile = config["train_images"];
    std::string trainingLabelsFile = config["train_labels"];
    std::string testImagesFile = config["test_images"];
    std::string testLabelsFile = config["test_labels"];


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
    network.addLayer(hiddenSize2, outputSize, Activation::identity, nullptr, true);

    int epochs = 10;
    int batchSize = 32;
    network.fit(trainImages, trainLabels, epochs, batchSize);

    double testAccuracy = network.evaluate(testImages, testLabels);
    std::cout << "Test accuracy: " << testAccuracy * 100.0 << "%" << std::endl;

    std::string modelFile = "mnist_fashion_model.bin";
    Serialization::serializeNetwork(network, modelFile);

    return 0;
}


//#include "utils/MNISTLoader.h"
//#include "nn/NeuralNetwork.h"
//#include "utils/Serialization.h"
//#include <iostream>
//#include <vector>
//#include <string>
//
//int main() {
//    std::string testImagesFile = "C:\\CLionProjects\\nn\\data\\fashion_mnist\\fashion-test-img";
//    std::string testLabelsFile = "C:\\CLionProjects\\nn\\data\\fashion_mnist\\fashion-test-labels";
//    std::string modelFile = "mnist_fashion_model.bin";
//
//    // Wczytywanie danych testowych
//    MNISTLoader loader;
//    std::vector<Matrix> testImages = loader.loadImages(testImagesFile);
//    std::vector<Matrix> testLabels = loader.loadLabels(testLabelsFile, 10);
//
//    // Deserializacja (wczytanie) wcześniej wytrenowanego modelu
//    NeuralNetwork loadedNetwork = Serialization::deserializeNetwork(modelFile);
//
//    // Ocena modelu na zbiorze testowym
//    double testAccuracy = loadedNetwork.evaluate(testImages, testLabels);
//    std::cout << "Test accuracy of loaded model: " << testAccuracy * 100.0 << "%" << std::endl;
//
//    return 0;
//}
