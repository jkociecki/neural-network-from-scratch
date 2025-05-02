#include "utils/MNISTLoader.h"
#include "nn/Matrix.h"
#include <fstream>
#include <iostream>
#include <cstdint>

std::vector<Matrix> MNISTLoader::loadImages(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    int32_t magic_number = 0, num_images = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);

    if (magic_number != 2051)
    {
        std::cerr << "Invalid magic number for images file: " << magic_number << std::endl;
        return {};
    }

    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::cout << "Loading " << num_images << " images (" << rows << "x" << cols << ")..." << std::endl;

    std::vector<Matrix> images;
    for (int i = 0; i < num_images; ++i)
    {
        Matrix image(rows * cols, 1);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image.at(r * cols + c, 0) = static_cast<double>(pixel) / 255.0;
            }
        }
        images.push_back(image);

        if ((i + 1) % 10000 == 0 || i == num_images - 1)
        {
            std::cout << "Loaded " << (i + 1) << " images..." << std::endl;
        }
    }

    return images;
}

std::vector<Matrix> MNISTLoader::loadLabels(const std::string& filename, int numClasses)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    int32_t magic_number = 0, num_items = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);

    if (magic_number != 2049)
    {
        std::cerr << "Invalid magic number for labels file: " << magic_number << std::endl;
        return {};
    }

    file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = __builtin_bswap32(num_items);

    std::cout << "Loading " << num_items << " labels..." << std::endl;

    std::vector<Matrix> labels;
    for (int i = 0; i < num_items; ++i)
    {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);

        Matrix oneHot = Matrix::zeros(numClasses, 1);
        oneHot.at(label, 0) = 1.0;
        labels.push_back(oneHot);

        if ((i + 1) % 10000 == 0 || i == num_items - 1)
        {
            std::cout << "Loaded " << (i + 1) << " labels..." << std::endl;
        }
    }

    return labels;
}

void MNISTLoader::splitData(const std::vector<Matrix>& allImages, const std::vector<Matrix>& allLabels,
                            std::vector<Matrix>& trainImages, std::vector<Matrix>& trainLabels,
                            std::vector<Matrix>& valImages, std::vector<Matrix>& valLabels,
                            double valRatio)
{
    size_t totalSize = allImages.size();
    size_t valSize = static_cast<size_t>(totalSize * valRatio);
    size_t trainSize = totalSize - valSize;

    trainImages.clear();
    trainLabels.clear();
    valImages.clear();
    valLabels.clear();

    trainImages.insert(trainImages.end(), allImages.begin(), allImages.begin() + trainSize);
    trainLabels.insert(trainLabels.end(), allLabels.begin(), allLabels.begin() + trainSize);

    valImages.insert(valImages.end(), allImages.begin() + trainSize, allImages.end());
    valLabels.insert(valLabels.end(), allLabels.begin() + trainSize, allLabels.end());

    std::cout << "Data split: " << trainSize << " training samples, "
              << valSize << " validation samples" << std::endl;
}
