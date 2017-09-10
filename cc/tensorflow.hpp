#ifndef DEEP_LEARNING_TENSORFLOW_HPP_
#define DEEP_LEARNING_TENSORFLOW_HPP_

#include <string>
#include <vector>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/standard_ops.h>

namespace tf = tensorflow;

#include "common.hpp"

// A simple neural network implementation using only full-connected neurals.
class TensorflowSimpleNetwork {
  public:
    TensorflowSimpleNetwork(const std::vector<Layer>& layers, int mini_batch_size);

    void Train(
        const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
        float weight_decay, float learning_rate, const std::vector<Case>* testing_data);

    float Evaluate(const std::vector<Case>& testing_data);

  private:
    const std::vector<Layer> layers_;
    const int mini_batch_size_;
    int input_size_;
    int output_classes_;
    tf::Scope scope_;
    tf::ClientSession session_;
    tf::ops::Placeholder inputs_;
    tf::ops::Placeholder labels_;
    tf::Output accuracy_;
};

#endif  // DEEP_LEARNING_TENSORFLOW_HPP_
