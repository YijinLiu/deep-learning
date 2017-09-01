#ifndef DEEP_LEARNING_TENSORFLOW_HPP_
#define DEEP_LEARNING_TENSORFLOW_HPP_

#include "common.hpp"

// A simple neural network implementation using only full-connected neurals.
class TensorflowSimpleNetwork {
  public:
    TensorflowSimpleNetwork(
        const std::vector<Layer>& layers, int mini_batch_size, float weight_decay,
        float learning_rate);

    void Train(
        const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
        const std::vector<Case>* testing_data);

    float Evaluate(const std::vector<Case>& testing_data);

  private:
    const std::vector<Layer> layers_;
    const int mini_batch_size_;
    int input_size_;
    int output_classes_;
};

#endif  // DEEP_LEARNING_TENSORFLOW_HPP_
