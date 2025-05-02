cmake_minimum_required(VERSION 3.27)
project(nn)

set(CMAKE_CXX_STANDARD 20)

# Include directories
include_directories(include)

# Source files
set(SOURCES
        main.cpp
        src/nn/Matrix.cpp
        src/nn/Layer.cpp
        src/nn/NeuralNetwork.cpp
        src/utils/MNISTLoader.cpp
)

# Header files
set(HEADERS
        include/nn/Matrix.h
        include/nn/Layer.h
        include/nn/NeuralNetwork.h
        include/nn/Activation.h
        include/utils/MNISTLoader.h
        include/optim/Optimizer.h
        src/optim/SGD.cpp
        include/optim/SGD.h
        src/optim/Adam.cpp
        include/optim/Adam.h
        predictor.cpp
        include/utils/Serialization.h
        src/utils/Serialization.cpp
)



# Add the executable
add_executable(nn ${SOURCES} ${HEADERS})

# Predictor executable
add_executable(predictor
        predictor.cpp
        src/nn/Matrix.cpp
        src/nn/Layer.cpp
        src/nn/NeuralNetwork.cpp
        src/optim/SGD.cpp
        src/optim/Adam.cpp
        src/utils/Serialization.cpp
)

target_include_directories(predictor PRIVATE include)
