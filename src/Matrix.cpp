#include "../include/Matrix.h"
#include <iostream>
#include <cassert>
#include <iomanip>
#include <random>


Matrix Matrix::genericOperator(const Matrix &other, const std::function<double(double, double)>& func) const
{
    assert(rows == other.rows && cols == other.cols);
    Matrix result(rows, cols);

    for(size_t y = 0; y < rows; y++)
    {
        for(size_t x = 0; x < cols; x++)
        {
            result.matrix[y][x] = func(matrix[y][x], other.matrix[y][x]);
        }
    }

    return result;
}

Matrix Matrix::apply(const std::function<double(double)>& func) const
{
    Matrix result(rows, cols);

    for (size_t y = 0; y < rows; y++)
    {
        for (size_t x = 0; x < cols; x++)
        {
            result.matrix[y][x] = func(matrix[y][x]);
        }
    }

    return result;
}

Matrix Matrix::dot(const Matrix& other) const
{
    assert(cols == other.rows);
    Matrix result(rows, other.cols, 0.0);

    for (size_t y = 0; y < rows; y++)
    {
        for (size_t x = 0; x < other.cols; x++)
        {
            for (size_t i = 0; i < cols; i++)
            {
                result.matrix[y][x] += matrix[y][i] * other.matrix[i][x];
            }
        }
    }

    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);

    for(size_t y = 0; y < rows; y++)
    {
        for(size_t x = 0; x < cols; x++)
        {
            result.matrix[x][y] = matrix[y][x];
        }
    }

    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, double min, double max)
{
    Matrix result(rows, cols);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(min, max);

    for (size_t y = 0; y < rows; y++)
    {
        for (size_t x = 0; x < cols; x++)
        {
            result.matrix[y][x] = dist(generator);
        }
    }

    return result;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    return genericOperator(other, [](double a, double b) { return a + b; });
}

Matrix Matrix::operator-(const Matrix &other) const
{
    return genericOperator(other, [](double a, double b) { return a - b; });
}

Matrix Matrix::operator*(const Matrix &other) const
{
    return genericOperator(other, [](double a, double b) { return a * b; });
}

Matrix Matrix::operator/(const Matrix &other) const
{
    return genericOperator(other, [](double a, double b) { return a / b; });
}

Matrix Matrix::operator+(double scalar) const
{
    return apply([scalar](double a) { return a + scalar; });
}

Matrix Matrix::operator-(double scalar) const
{
    return apply([scalar](double a) { return a - scalar; });
}

Matrix Matrix::operator*(double scalar) const
{
    return apply([scalar](double a) { return a * scalar; });
}

Matrix Matrix::operator/(double scalar) const
{
    assert(scalar != 0.0);
    return apply([scalar](double a) { return a / scalar; });
}

void Matrix::print(int precision) const
{
    for (const auto& row : matrix)
    {
        for (double val : row)
        {
            std::cout << std::fixed << std::setprecision(precision) << val << " ";
        }
        std::cout << "\n";
    }
}



Matrix Matrix::ones(size_t rows, size_t cols)
{
    return {rows, cols, 1.0};
}

Matrix Matrix::zeros(size_t rows, size_t cols)
{
    return {rows, cols, 0.0};
}
