#include "simple_network.hpp"

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <iomanip>
#include <random>

#include <glog/logging.h>

SimpleNetwork::SimpleNetwork(const std::vector<Layer>& layers, size_t mini_batch_size)
    : layers_(layers), mini_batch_size_(mini_batch_size) {
    // Check layers.
    CHECK_GE(layers.size(), 2);
    const auto& input_layer = layers.front();
    if (input_layer.activation != ActivationFunc::Identity) {
        LOG(FATAL)
            << "Input layer's activation function needs to be identity, found "
            << static_cast<int>(input_layer.activation);
    }
    input_size_ = input_layer.num_neurons;
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
    output_classes_ = output_layer.num_neurons;

    // Init parameters.
    biases_.resize(layers.size() - 1);
    std::default_random_engine generator;
    weights_.resize(layers.size() - 1);
    for (int i = 1; i < layers.size(); i++) {
        const int rows = layers[i-1].num_neurons;
        const int cols = layers[i].num_neurons;
        auto& weight = weights_[i-1];
        weight.resize(rows, cols);
        std::normal_distribution<float> distribution(0.f, 2.f / (rows + cols));
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                weight(r, c) = distribution(generator);
            }
        }
    }
    for (int i = 1; i < layers.size(); i++) Randomize(biases_[i-1], layers[i].num_neurons);
}

void SimpleNetwork::Train(
    const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
    float weight_decay, float learning_rate, const std::vector<Case>* testing_data) {
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
        int corrects = 0;
        for (size_t k = 0; k <= n - mini_batch_size_; k += mini_batch_size_) {
            corrects += UpdateMiniBatch(training_data, indices, k, weight_decay, learning_rate);
        }
        VLOG(0) << "Epoch " << e + 1 << " training accuracy: "
            << std::setprecision(4) << float(corrects) / n << "(" << corrects << "/" << n << ").";
        if (testing_data != nullptr) {
            const size_t corrects = Evaluate(*testing_data);
            LOG(INFO) << "Epoch " << e + 1 << " testing accuracy: "
                << std::setprecision(4) << float(corrects) / testing_data->size()
                << "(" << corrects << "/" << testing_data->size() << ").";
        }
    }
}

size_t SimpleNetwork::Evaluate(const std::vector<Case>& testing_data) {
    size_t corrects = 0;
    Matrix inputs(mini_batch_size_, input_size_);
    for (size_t k = 0; k <= testing_data.size() - mini_batch_size_; k += mini_batch_size_) {
        for (int i = 0; i < mini_batch_size_; i++) inputs.row(i) = testing_data[k + i].first;
        const auto a = FeedForward(inputs);
        for (int i = 0; i < mini_batch_size_; i++) {
            int predict;
            MAX_VAL_INDEX(a.row(i), predict);
            if (predict == testing_data[k + i].second) corrects++;
        }
    }
    return corrects;
}

inline Matrix SimpleNetwork::Z(int layer, const Matrix& a) const {
    Matrix z = a * weights_[layer-1];
#ifdef USE_EIGEN
    z.rowwise() += biases_[layer-1];
#elif defined(USE_ARMADILLO)
    z.each_row() += biases_[layer-1];
#endif
    return z;
}

inline Matrix SimpleNetwork::Activation(int layer, const Matrix& z) const {
    Matrix ret;
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

inline Matrix SimpleNetwork::ActivationDerivative(int layer, const Matrix& z) const {
    Matrix ret;
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

inline Matrix SimpleNetwork::FeedForward(const Matrix& inputs) const {
    Matrix a = inputs;
    for (int i = 1; i < layers_.size(); i++) {
        a = Activation(i, Z(i, a));
    }
    return a;
}

inline int SimpleNetwork::UpdateMiniBatch(const std::vector<Case>& training_data,
                                          const std::vector<int> indices, int start, 
                                          float weight_decay, float learning_rate) {
    // Initialize param deltas as zero.
    std::vector<Matrix> weights_delta(weights_.size());
    for (int i = 0; i < weights_.size(); i++) Zeros(weights_delta[i], weights_[i]);
    std::vector<Vector> biases_delta(biases_.size());
    for (int i = 0; i < biases_.size(); i++) Zeros(biases_delta[i], biases_[i]);

    // Construct inputs and labels.
    Matrix inputs(mini_batch_size_, input_size_);
    std::vector<int> labels;
    labels.reserve(mini_batch_size_);
    for (int i = 0; i < mini_batch_size_; i++) {
        const int index = indices[start + i];
        const auto& sample = training_data[index];
        inputs.row(i) = sample.first;
        labels.push_back(sample.second);
    }

    // Update params.
    const int corrects = BackPropagate(inputs, labels, weights_delta, biases_delta);
    const float multiplier = learning_rate / mini_batch_size_;
    for (int i = 0; i < biases_.size(); i++) {
#ifdef USE_EIGEN
        biases_[i].noalias() -= biases_delta[i] * multiplier;
#else
        biases_[i] -= biases_delta[i] * multiplier;
#endif
    }
    for (int i = 0; i < weights_.size(); i++) {
        weights_[i] *= weight_decay;
#ifdef USE_EIGEN
        weights_[i].noalias() -= weights_delta[i] * multiplier;
#else
        weights_[i] -= weights_delta[i] * multiplier;
#endif
    }
    return corrects;
}

inline int SimpleNetwork::BackPropagate(const Matrix& inputs, const std::vector<int>& labels,
                                        std::vector<Matrix>& weights_delta,
                                        std::vector<Vector>& biases_delta) const {
    const int n = layers_.size();
    std::vector<Matrix> zs(n - 1);
    std::vector<Matrix> as(n);
    Matrix a = inputs;
    as[0] = a;
    for (int i = 1; i < n; i++) {
        const Matrix z = Z(i, a);
        zs[i-1] = z;
        as[i] = a = Activation(i, z);
    }
    // When the output layer's activation is sigmoid, we use cross entropy cost.
    // When the output layer's activation is softmax, we use max likelihood.
    // The gradient of z at output layer is "a - y" for both cases.
    Matrix delta = a;
    int corrects = 0;
    for (int r = 0; r < labels.size(); r++) {
        int predict;
        MAX_VAL_INDEX(a.row(r), predict);
        const int truth = labels[r];
        if (predict == truth) corrects++;
        delta(r, truth) -= 1.f;
    }
    biases_delta[n-2] += MAT_COL_SUM(delta);
    weights_delta[n-2] += MAT_T(as[n-2]) * delta;
    for (int i = n - 2; i > 0; i--) {
        delta *= MAT_T(weights_[i]);
        MAT_CWISE_MUL(delta, ActivationDerivative(i, zs[i-1]));
#ifdef USE_EIGEN
        biases_delta[i-1].noalias() += MAT_COL_SUM(delta);
        weights_delta[i-1].noalias() += MAT_T(as[i-1]) * delta;
#else
        biases_delta[i-1] += MAT_COL_SUM(delta);
        weights_delta[i-1] += MAT_T(as[i-1]) * delta;
#endif
    }
    return corrects;
}
