#ifndef DEEP_LEARNING_MNIST_HPP_
#define DEEP_LEARNING_MNIST_HPP_

#include <utility>
#include <vector>

#include <Eigen/Dense>

std::vector<std::pair<Eigen::VectorXf, int>> LoadMNISTData(const char* dir, const char* name);

#endif  // DEEP_LEARNING_MNIST_HPP_
