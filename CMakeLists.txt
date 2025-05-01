cmake_minimum_required(VERSION 3.27)
project(nn)

set(CMAKE_CXX_STANDARD 20)

add_executable(nn main.cpp
        Matrix.cpp
        Matrix.h
        Matrix.h
        Layer.cpp
        Layer.h)
