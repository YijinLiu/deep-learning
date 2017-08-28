#ifndef DEEP_LEARNING_COMMON_HPP_
#define DEEP_LEARNING_COMMON_HPP_

#include <utility>

#include "linear_algebra.hpp"

enum class ActivationFunc {
    Identity,
    ReLU,
    Sigmoid,
    SoftMax
};

struct Layer {
    Layer(int num_neurons, ActivationFunc activation) : num_neurons(num_neurons),
    activation(activation) {}
    int num_neurons;
    ActivationFunc activation;
};

typedef std::pair<Vector, int> Case;

#endif  // DEEP_LEARNING_COMMON_HPP_
