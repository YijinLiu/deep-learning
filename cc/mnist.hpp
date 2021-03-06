#ifndef DEEP_LEARNING_MNIST_HPP_
#define DEEP_LEARNING_MNIST_HPP_

#include <utility>
#include <vector>

#include "common.hpp"

std::vector<Case> LoadMNISTData(const char* dir, const char* name);

#endif  // DEEP_LEARNING_MNIST_HPP_
