//
// Created by Jędrzej on 2025-05-01.
//

#ifndef NN_MATRIX_H
#define NN_MATRIX_H
#pragma once
#include <iostream>
#include <vector>
#include <functional>


class Matrix
{
public:
    std::vector<std::vector<double>> matrix;
    size_t rows, cols;


    Matrix(size_t r, size_t c, double val = 0.0)
        : rows(r), cols(c), matrix(r, std::vector<double>(c, val)) {}


    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator/(const Matrix& other) const;
    Matrix operator+(const double scalar) const;
    Matrix operator-(const double scalar) const;
    Matrix operator*(const double scalar) const;
    Matrix operator/(const double scalar) const;
    Matrix apply(const std::function<double(double)>& func) const;
    Matrix dot(const Matrix& other) const;
    Matrix transpose() const;
    void print(int precision) const;



private:
    Matrix genericOperator(const Matrix& other, const std::function<double(double, double)>& func) const;

};


#endif //NN_MATRIX_H
