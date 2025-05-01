cmake_minimum_required(VERSION 3.27)
project(nn)

set(CMAKE_CXX_STANDARD 20)

add_executable(nn main.cpp
        src/Matrix.cpp
        include/Matrix.h
        include/Matrix.h
        src/Layer.cpp
        include/Layer.h
        src/NeuralNetwork.cpp
        include/NeuralNetwork.h)
