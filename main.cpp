#include <iostream>
#include "include/Matrix.h"
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));

    Matrix matrix1(2, 2);
    matrix1 = matrix1.apply([](double) {
        return static_cast<double>(1 + rand() % 10);
    });

    matrix1.print(0);

    std::cout << "\n";

    std::srand(std::time(nullptr));

    Matrix matrix2(2, 2);
    matrix2 = matrix2.apply([](double) {
        return static_cast<double>(1 + rand() % 10);
    });

    matrix2.print(0);
    std::cout << "\n";

    Matrix result = matrix1.dot(matrix2);
    result.print(0);

    return 0;
}