#ifndef DEEP_LEARNING_SIMPLE_NETWORK_HPP_
#define DEEP_LEARNING_SIMPLE_NETWORK_HPP_

#include <inttypes.h>

#include <utility>
#include <vector>

#include "common.hpp"

// A simple neural network implementation using only full-connected neurals.
class SimpleNetwork {
  public:
    SimpleNetwork(const std::vector<Layer>& layers, size_t mini_batch_size);

    void Train(
        const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
        float weight_decay, float learning_rate, const std::vector<Case>* testing_data);

    size_t Evaluate(const std::vector<Case>& testing_data);

  private:
    inline Matrix Z(int layer, const Matrix& z) const;
    inline Matrix Activation(int layer, const Matrix& z) const;
    inline Matrix ActivationDerivative(int layer, const Matrix& z) const;

    inline Matrix FeedForward(const Matrix& inputs) const;

    // Returns number of correct predicts.
    inline int BackPropagate(const Matrix& inputs, const std::vector<int>& labels,
                             std::vector<Matrix>& weights_delta,
                             std::vector<Vector>& biases_delta) const;

    // Returns number of correct predicts.
    inline int UpdateMiniBatch(
        const std::vector<Case>& training_data, const std::vector<int> indices, int start, 
        float weight_decay, float learning_rate);

    const std::vector<Layer> layers_;
    const size_t mini_batch_size_;
    int input_size_;
    int output_classes_;
    std::vector<Matrix> weights_;
    std::vector<Vector> biases_;
};

#endif  // DEEP_LEARNING_FEEDFORWARD_NETWORK_HPP_
