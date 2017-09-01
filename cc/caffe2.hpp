#ifndef DEEP_LEARNING_CAFFE2_HPP_
#define DEEP_LEARNING_CAFFE2_HPP_

#include <caffe2/core/workspace.h>

#include "common.hpp"

// A simple neural network implementation using only full-connected neurals.
class Caffe2SimpleNetwork {
  public:
    Caffe2SimpleNetwork(
        const std::vector<Layer>& layers, int mini_batch_size, float weight_decay,
        float learning_rate);

    void Train(
        const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
        const std::vector<Case>* testing_data);

    float Evaluate(const std::vector<Case>& testing_data);

  private:
    std::string AddLayers(caffe2::NetDef* net, bool train) const;
    float Accuracy() const;
    void AddInput(int i, const Case& c);

    const std::vector<Layer> layers_;
    const int mini_batch_size_;
    int input_size_;
    int output_classes_;
    // mini_batch_size_ * input_size_ items, not owned.
    float* inputs_data_;
    // mini_batch_size_ items, not owned.
    int* labels_data_;
    bool expands_label_;
    // mini_batch_size_ * input_size_ items, not owned.
    float* expanded_labels_data_;
    caffe2::Workspace workspace_;
    std::unique_ptr<caffe2::NetBase> train_net_;
    std::unique_ptr<caffe2::NetBase> predict_net_;
};

#endif  // DEEP_LEARNING_CAFFE2_HPP_
