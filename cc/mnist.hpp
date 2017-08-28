#ifndef DEEP_LEARNING_MNIST_HPP_
#define DEEP_LEARNING_MNIST_HPP_

#include <utility>
#include <vector>

#include "linear_algebra.hpp"

std::vector<std::pair<Vector, int>> LoadMNISTData(const char* dir, const char* name);

#endif  // DEEP_LEARNING_MNIST_HPP_
