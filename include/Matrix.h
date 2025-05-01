
#ifndef NN_MATRIX_H
#define NN_MATRIX_H
#pragma once
#include <iostream>
#include <vector>
#include <functional>


class Matrix
{

public:


    Matrix() : rows(0), cols(0) {}
    Matrix(size_t r, size_t c, double val = 0.0)
        : rows(r), cols(c), matrix(r, std::vector<double>(c, val)) {}

    static Matrix random(size_t rows, size_t cols, double min, double max);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix zeros(size_t rows, size_t cols);

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator/(const Matrix& other) const;
    Matrix operator+(double scalar) const;
    Matrix operator-(double scalar) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    Matrix apply(const std::function<double(double)>& func) const;
    [[nodiscard]] Matrix dot(const Matrix& other) const;
    [[nodiscard]] Matrix transpose() const;
    [[nodiscard]] double at(size_t row, size_t col) const { return matrix[row][col]; }
    [[nodiscard]] size_t getRows() const { return rows; }
    [[nodiscard]] size_t getCols() const { return cols; }
    void print(int precision) const;



private:
    std::vector<std::vector<double>> matrix;
    size_t rows, cols;
    Matrix genericOperator(const Matrix& other, const std::function<double(double, double)>& func) const;

};


#endif //NN_MATRIX_H
