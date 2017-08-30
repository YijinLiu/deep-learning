#ifndef DEEP_LEARNING_CAFFE2_HPP_
#define DEEP_LEARNING_CAFFE2_HPP_

#include "common.hpp"

class Caffe2FeedForwardNetwork {
  public:
    Caffe2FeedForwardNetwork(const std::vector<Layer>& layers, float weight_decay);

    void StochasticGradientDescent(
        const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
        size_t mini_batch_size, float learning_rate, const std::vector<Case>* testing_data);

    size_t Evaluate(const std::vector<Case>& testing_data);

  private:
    std::string AddLayers(caffe2::NetDef* net, bool train) const;

    const std::vector<Layer> layers_;
    const float weight_decay_;
};

#endif
