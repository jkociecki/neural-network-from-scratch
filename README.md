# Neural Network from Scratch

This project implements a fully custom neural network in C++, trained and evaluated on the MNIST and Fashion-MNIST datasets. The goal was to build every component from scratch — including matrix operations, layers, activations, optimizers, and serialization — without relying on external ML frameworks.

## Model Overview

The neural network supports:

* Multiple layers (dense/fully connected)
* ReLU and softmax activations
* SGD and Adam optimizers
* Serialization to binary format for model saving/loading

## Dataset and Performance

Two datasets were used for training and evaluation:

* **MNIST** (handwritten digits):

  * Achieved \~94% accuracy on the test set
* **Fashion-MNIST** (clothing items):

  * Achieved \~89% accuracy on the test set

Data is loaded from `.idx` binary files directly via custom C++ parsing routines.

## Visualizing Predictions

The project includes a set of scripts that generate sample input images and visualize predictions made by the trained model. These images allow you to see how the model classifies individual digits or clothing items. Predictions are overlayed on the images for clarity.

All example predictions shown were generated using the actual trained model and real test data. They are located in the `scripts` directory and demonstrate the model's inference behavior.
![mnist_predictions_grid.png](scripts%2Fimgs%2Fmnist_predictions_grid.png)
![fashion_mnist_predictions_grid.png](scripts%2Fimgs%2Ffashion_mnist_predictions_grid.png)


## Running the Project

You can then run the model training or prediction:

```bash
./nn        # Run to train the model
./predictor # Created for visualization purposes
            # Compile it and use it in python scripts
```

## GUI
The project includes a graphical interface that allows users to browse through test images and observe how the model classifies each one, along with probability distributions and prediction history. In the example shown, the model incorrectly predicted the digit "8" instead of the correct label "3" — a mistake that’s visually understandable given the similar shape of the digit in this case.
![missclas.jpg](scripts%2Fimgs%2Fmissclas.jpg)

## Project Structure (Brief)

* `neural_network/` — source code for training and inference
* `data/` — raw MNIST and Fashion-MNIST dataset files
* `scripts/` — prediction visualizations and helper scripts


