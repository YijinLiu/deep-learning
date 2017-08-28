#include "caffe2.hpp"

Caffe2FeedForwardNetwork::Caffe2FeedForwardNetwork(
    const std::vector<Layer>& layers, float weight_decay) : workspace_("feedforward") {
}


void Caffe2FeedForwardNetwork::StochasticGradientDescent(
    const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
    size_t mini_batch_size, float learning_rate, const std::vector<Case>* testing_data) {
    caffe2::NetDef init_net, predict_net;
    init_net.set_name("feedforward_init");
    predict_net.set_name("feedforward_predict");
}
