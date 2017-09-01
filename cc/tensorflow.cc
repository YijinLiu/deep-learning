#include "tensorflow.hpp"

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/ops/standard_ops.h>

TensorflowSimpleNetwork::TensorflowSimpleNetwork(
    const std::vector<Layer>& layers, int mini_batch_size, float weight_decay, float learning_rate)
    : layers_(layers), mini_batch_size_(mini_batch_size) {
}

void TensorflowSimpleNetwork::Train(
    const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
    const std::vector<Case>* testing_data) {
}

float TensorflowSimpleNetwork::Evaluate(const std::vector<Case>& testing_data) {
}
