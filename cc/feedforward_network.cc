#include "feedforward_network.hpp"

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <iomanip>

#include <glog/logging.h>

namespace {

Eigen::VectorXf Sigmoid(const Eigen::VectorXf& z) {
    Eigen::VectorXf s(z.size());
    for (int i = 0; i < z.size(); i++) s(i) = 1.f / (1.f + expf(-z(i)));
    return s;
}

Eigen::VectorXf SigmoidDerivative(const Eigen::VectorXf& z) {
    const Eigen::VectorXf s = Sigmoid(z);
    return s.cwiseProduct(Eigen::VectorXf::Constant(z.size(), 1.f) - s);
}

}  // namespace

FeedForwardNetwork::FeedForwardNetwork(const std::vector<size_t> layer_sizes)
    : num_layers_(layer_sizes.size()) {
    biases_.resize(num_layers_ - 1);
    for (int i = 1; i < num_layers_; i++) {
        biases_[i - 1].resize(layer_sizes[i]);
        biases_[i - 1].setRandom();
    }
    weights_.resize(num_layers_ - 1);
    for (int i = 1; i < num_layers_; i++) {
        weights_[i - 1].resize(layer_sizes[i], layer_sizes[i - 1]);
        weights_[i - 1].setRandom();
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
                << std::setprecision(2) << float(corrects) / testing_data->size()
                << "(" << corrects << "/" << testing_data->size() << ").";
        }
    }
}

Eigen::VectorXf FeedForwardNetwork::FeedForward(const Eigen::VectorXf& x) const {
    Eigen::VectorXf a = x;
    for (int i = 0; i < num_layers_ - 1; i++) a = Sigmoid(weights_[i] * a + biases_[i]);
    return a;
}

size_t FeedForwardNetwork::Evaluate(const std::vector<Case>& testing_data) {
    size_t corrects = 0;
    for (const auto& sample : testing_data) {
        int index;
        FeedForward(sample.first).maxCoeff(&index);
        if (index == sample.second) corrects++;
    }
    return corrects;
}

void FeedForwardNetwork::UpdateMiniBatch(const std::vector<Case>& training_data,
                                         std::vector<int>::const_iterator begin,
                                         std::vector<int>::const_iterator end,
                                         float learning_rate) {
    // Initialize deltas as zero.
    std::vector<Eigen::VectorXf> biases_delta(biases_.size());
    for (int i = 0; i < biases_.size(); i++) {
        biases_delta[i].resize(biases_[i].size());
        biases_delta[i].fill(0.f);
    }
    std::vector<Eigen::MatrixXf> weights_delta(weights_.size());
    for (int i = 0; i < weights_.size(); i++) {
        weights_delta[i].resize(weights_[i].rows(), weights_[i].cols());
        weights_delta[i].fill(0.f);
    }
    for (auto iter = begin; iter != end; iter++) {
        const auto& sample = training_data[*iter];
        BackPropagate(sample.first, sample.second, biases_delta, weights_delta);
    }
    const float multiplier = learning_rate / (end - begin);
    for (int i = 0; i < biases_.size(); i++) biases_[i] -= biases_delta[i] * multiplier;
    for (int i = 0; i < weights_.size(); i++) weights_[i] -= weights_delta[i] * multiplier;
}

void FeedForwardNetwork::BackPropagate(const Eigen::VectorXf& x, int y,
                                       std::vector<Eigen::VectorXf>& biases_delta,
                                       std::vector<Eigen::MatrixXf>& weights_delta) {
    std::vector<Eigen::VectorXf> zs;
    std::vector<Eigen::VectorXf> as;
    Eigen::VectorXf a = x;
    as.push_back(a);
    const int n = num_layers_ - 1;
    for (int i = 0; i < n; i++) {
        const Eigen::VectorXf z = weights_[i] * a + biases_[i];
        zs.push_back(z);
        a = Sigmoid(z);
        as.push_back(a);
    }
    Eigen::VectorXf delta = a;
    delta[y] -= 1.f;
    for (int i = n - 1; i >= 0; i--) {
        delta = delta.cwiseProduct(SigmoidDerivative(zs[i]));
        biases_delta[i] += delta;
        weights_delta[i] += delta * as[i].transpose();
        if (i > 0) delta.applyOnTheLeft(weights_[i].transpose());
    }
}
