#include "feedforward_network.hpp"

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <iomanip>
#include <random>

#include <glog/logging.h>

FeedForwardNetwork::FeedForwardNetwork(const std::vector<Layer>& layers, float weight_decay)
    : layers_(layers), weight_decay_(weight_decay) {
    // Check layers.
    CHECK_GE(layers.size(), 2);
    const auto& input_layer = layers.front();
    if (input_layer.activation != ActivationFunc::Identity) {
        LOG(FATAL)
            << "Input layer's activation function needs to be identity, found "
            << static_cast<int>(input_layer.activation);
    }
    const auto& output_layer = layers.back();
    switch (output_layer.activation) {
        case ActivationFunc::Sigmoid:
        case ActivationFunc::SoftMax:
            break;
        default:
            LOG(FATAL)
                << "Output layer's activation function needs to be sigmoid or softmax, found "
                << static_cast<int>(output_layer.activation);
    }

    // Init parameters.
    biases_.resize(layers.size() - 1);
    for (int i = 1; i < layers.size(); i++) {
        Randomize(biases_[i-1], layers[i].num_neurons);
    }
    std::default_random_engine generator;
    weights_.resize(layers.size() - 1);
    for (int i = 1; i < layers.size(); i++) {
        const int rows = layers[i].num_neurons;
        const int cols = layers[i-1].num_neurons;
        auto& weight = weights_[i-1];
        weight.resize(rows, cols);
        std::normal_distribution<float> distribution(0.f, 2.f / (rows + cols));
        for (int j = 0; j < rows * cols; j++) weight[j] = distribution(generator);
    }
}

void FeedForwardNetwork::StochasticGradientDescent(
    const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
    size_t mini_batch_size, float learning_rate, const std::vector<Case>* testing_data) {
    size_t n = std::min(training_data.size(), num_samples_per_epoch);
    std::vector<int> indices(training_data.size());
    for (int i = 0; i < training_data.size(); i++) indices[i] = i;
    srand48(time(NULL));
    for (int e = 0; e < epochs; e++) {
        // Random shuffle.
        for (size_t i = 0; i < n; i++) {
            const size_t step = lrand48() % (training_data.size() - i);
            if (step > 0) std::swap(indices[i], indices[i + step]);
        }
        for (size_t k = 0; k <= n - mini_batch_size; k += mini_batch_size) {
            UpdateMiniBatch(training_data, indices.begin() + k, indices.begin() + k + mini_batch_size,
                            learning_rate);
        }
        if (testing_data != nullptr) {
            const size_t corrects = Evaluate(*testing_data);
            VLOG(1) << "Epoch " << e + 1 << ": "
                << std::setprecision(4) << float(corrects) / testing_data->size()
                << "(" << corrects << "/" << testing_data->size() << ").";
        }
    }
}

Vector FeedForwardNetwork::Activation(int layer, const Vector& z) const {
    Vector ret;
    switch (layers_[layer].activation) {
        case ActivationFunc::Identity:
            ret = z;
            break;
        case ActivationFunc::ReLU:
            ret = ReLU(z);
            break;
        case ActivationFunc::Sigmoid:
            ret = Sigmoid(z);
            break;
        case ActivationFunc::SoftMax:
            ret = SoftMax(z);
            break;
        default:
            LOG(FATAL)
                << "Unknown activation function: "
                << static_cast<int>(layers_[layer].activation);
    }
    return ret;
}

Vector FeedForwardNetwork::ActivationDerivative(int layer, const Vector& z) const {
    Vector ret;
    switch (layers_[layer].activation) {
        case ActivationFunc::Identity:
            Ones(ret, z);
            break;
        case ActivationFunc::ReLU:
            ret = ReLUDerivative(z);
            break;
        case ActivationFunc::Sigmoid:
            ret = SigmoidDerivative(z);
            break;
        case ActivationFunc::SoftMax:
            LOG(FATAL) << "SoftMax should only be used in output layer!";
            break;
        default:
            LOG(FATAL)
                << "Unknown activation function: "
                << static_cast<int>(layers_[layer].activation);
    }
    return ret;
}

Vector FeedForwardNetwork::FeedForward(const Vector& x) const {
    Vector a = x;
    for (int i = 1; i < layers_.size(); i++) {
        a = Activation(i, weights_[i-1] * a + biases_[i-1]);
    }
    return a;
}

size_t FeedForwardNetwork::Evaluate(const std::vector<Case>& testing_data) {
    size_t corrects = 0;
    for (const auto& sample : testing_data) {
        if (IndexMax(FeedForward(sample.first)) == sample.second) corrects++;
    }
    return corrects;
}

void FeedForwardNetwork::UpdateMiniBatch(const std::vector<Case>& training_data,
                                         std::vector<int>::const_iterator begin,
                                         std::vector<int>::const_iterator end,
                                         float learning_rate) {
    // Initialize deltas as zero.
    std::vector<Vector> biases_delta(biases_.size());
    for (int i = 0; i < biases_.size(); i++) {
        Zeros(biases_delta[i], biases_[i]);
    }
    std::vector<Matrix> weights_delta(weights_.size());
    for (int i = 0; i < weights_.size(); i++) {
        Zeros(weights_delta[i], weights_[i]);
    }
    for (auto iter = begin; iter != end; iter++) {
        const auto& sample = training_data[*iter];
        BackPropagate(sample.first, sample.second, biases_delta, weights_delta);
    }
    const float multiplier = learning_rate / (end - begin);
    for (int i = 0; i < biases_.size(); i++) {
        biases_[i] -= biases_delta[i] * multiplier;
    }
    for (int i = 0; i < weights_.size(); i++) {
        weights_[i] *= weight_decay_;
        weights_[i] -= weights_delta[i] * multiplier;
    }
}

void FeedForwardNetwork::BackPropagate(const Vector& x, int y,
                                       std::vector<Vector>& biases_delta,
                                       std::vector<Matrix>& weights_delta) {
    const int n = layers_.size();
    std::vector<Vector> zs(n - 1);
    std::vector<Vector> as(n);
    Vector a = x;
    as[0] = a;
    for (int i = 1; i < n; i++) {
        const Vector z = weights_[i-1] * a + biases_[i-1];
        zs[i-1] = z;
        as[i] = a = Activation(i, z);
    }
    // When the output layer's activation is sigmoid, we use cross entropy cost.
    // When the output layer's activation is softmax, we use max likelihood.
    // The gradient of z at output layer is "a - y" for both cases.
    Vector delta = a;
    delta[y] -= 1.f;
    biases_delta[n-2] += delta;
    weights_delta[n-2] += delta * Transpose(as[n-2]);
    for (int i = n - 2; i > 0; i--) {
        ApplyOnLeft(delta, Transpose(weights_[i]));
        ElemWiseMul(delta, ActivationDerivative(i, zs[i-1]));
        biases_delta[i-1] += delta;
        weights_delta[i-1] += delta * Transpose(as[i-1]);
    }
}
