#ifndef DEEP_LEARNING_CAFFE2_HPP_
#define DEEP_LEARNING_CAFFE2_HPP_

#include <caffe2/core/workspace.h>

#include "common.hpp"

class Caffe2FeedForwardNetwork {
  public:
    Caffe2FeedForwardNetwork(
        const std::vector<Layer>& layers, int mini_batch_size, float learning_rate);

    void Train(
        const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
        const std::vector<Case>* testing_data);

    float Evaluate(const std::vector<Case>& testing_data);

  private:
    std::string AddLayers(caffe2::NetDef* net, bool train) const;

    float Accuracy() const;

    const std::vector<Layer> layers_;
    const int mini_batch_size_;
    caffe2::Workspace workspace_;
    std::unique_ptr<caffe2::NetBase> train_net_;
    std::unique_ptr<caffe2::NetBase> predict_net_;
};

#endif
