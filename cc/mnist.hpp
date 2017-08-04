#ifndef DEEP_LEARNING_MNIST_HPP_
#define DEEP_LEARNING_MNIST_HPP_

#include <utility>
#include <vector>

#include <armadillo>

std::vector<std::pair<arma::Col<float>, int>> LoadMNISTData(
    const char* dir, const char* name);

#endif  // DEEP_LEARNING_MNIST_HPP_
