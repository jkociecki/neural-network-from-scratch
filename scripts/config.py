import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MNIST_TEST_IMAGES = os.path.join(BASE_DIR, '..', 'data', 'mnist','t10k-images.idx3-ubyte')
MNIST_TEST_LABELS = os.path.join(BASE_DIR, '..', 'data', 'mnist','t10k-labels.idx1-ubyte')

FASHION_MNIST_TRAIN_IMAGES = os.path.join(BASE_DIR, '..', 'data', 'fashion_mnist', 'fashion-train-img')
FASHION_MNIST_TRAIN_LABELS = os.path.join(BASE_DIR, '..', 'data', 'fashion_mnist', 'fashion-train-labels')
FASHION_MNIST_TEST_IMAGES = os.path.join(BASE_DIR, '..', 'data', 'fashion_mnist', 'fashion-test-img')
FASHION_MNIST_TEST_LABELS = os.path.join(BASE_DIR, '..', 'data', 'fashion_mnist', 'fashion-test-labels')

PREDICTOR_EXE = os.path.join(BASE_DIR, '..', 'neural_network', 'cmake-build-debug', 'predictor.exe')

FASHION_MNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
] 