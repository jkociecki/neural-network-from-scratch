//
// Created by Jędrzej on 2025-05-01.
//

#ifndef NN_MNISTLOADER_H
#define NN_MNISTLOADER_H

#include <vector>
#include "nn/Matrix.h"

class MNISTLoader
{
public:
    static std::vector<Matrix> loadImages(const std::string& filename);

    static std::vector<Matrix> loadLabels(const std::string& filename, int numClasses = 10);

    static void splitData(const std::vector<Matrix>& allImages, const std::vector<Matrix>& allLabels,
                          std::vector<Matrix>& trainImages, std::vector<Matrix>& trainLabels,
                          std::vector<Matrix>& valImages, std::vector<Matrix>& valLabels,
                          double valRatio = 0.2);
};

#endif //NN_MNISTLOADER_H
