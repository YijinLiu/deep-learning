#ifndef DEEP_LEARNING_COMMON_HPP_
#define DEEP_LEARNING_COMMON_HPP_

#include <string>
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

inline std::string Sprintf(const char* fmt, ...) {
    char buf[100];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

inline std::string LayerW(int l) { return Sprintf("l%d_w", l); }

inline std::string LayerB(int l) { return Sprintf("l%d_b", l); }

inline std::string LayerZ(int l) { return Sprintf("l%d_z", l); }

inline std::string LayerA(int l) { return Sprintf("l%d_a", l); }

#endif  // DEEP_LEARNING_COMMON_HPP_
