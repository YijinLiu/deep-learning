#include "caffe2.hpp"

#include <stdio.h>

#include <string>
#include <vector>

namespace {

caffe2::Argument* AddOpArg(caffe2::OperatorDef& op, const std::string& name) {
    auto* arg = op.add_arg();
    arg->set_name(name);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef& op, const std::string& name, int value) {
    auto* arg = AddOpArg(op, name);
    arg->set_i(value);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef& op, const std::string& name, float value) {
    auto* arg = AddOpArg(op, name);
    arg->set_f(value);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef& op, const std::string& name,
                           const std::string& value) {
    auto* arg = AddOpArg(op, name);
    arg->set_s(value);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef& op, const std::string& name,
                           std::vector<int> values) {
    auto* arg = AddOpArg(op, name);
    for (auto value : values) arg->add_ints(value);
    return arg;
}


caffe2::Argument* AddOpArg(caffe2::OperatorDef& op, const std::string& name,
                           std::vector<float> values) {
    auto* arg = AddOpArg(op, name);
    for (auto value : values) arg->add_floats(value);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef& op, const std::string& name,
                           const std::vector<std::string>& values) {
    auto* arg = AddOpArg(op, name);
    for (auto value : values) arg->add_strings(value);
    return arg;
}

caffe2::OperatorDef* AddOp(caffe2::NetDef* net, const std::string& name,
                           const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs) {
    auto* op = net->add_op();
    op->set_type(name);
    for (const auto& input : inputs) op->add_input(input);
    for (const auto& output : outputs) op->add_output(output);
    return op;
}

caffe2::OperatorDef* AddXavierFillOp(caffe2::NetDef* net, const std::vector<int>& shape,
                                     const std::string& name) {
    auto* op = AddOp(net, "XavierFill", {}, {name});
    AddOpArg(op, "shape", shape);
    return op;
}

}  // namespace

Caffe2FeedForwardNetwork::Caffe2FeedForwardNetwork(
    const std::vector<Layer>& layers, float weight_decay) : workspace_("feedforward") {
    // Check layers.
    CHECK_GE(layers.size(), 2);
    const auto& input_layer = layers.front();
    if (input_layer.activation != ActivationFunc::Identity) {
        LOG(FATAL)
            << "Input layer's activation function needs to be identity, found "
            << static_cast<int>(input_layer.activation);
    }
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

    caffe2::NetDef init_net;
    init_net.set_name("init");
    char name_buf[100];
    for (int i = 1; i < layers.size(); i++) {
        const int rows = layers[i].num_neurons;
        const int cols = layers[i-1].num_neurons;
        snprintf(name_buf, sizeof(name_buf), "l%d_w", i);
        AddXavierFillOp(&init_net, {rows, cols}, name_buf);
        snprintf(name_buf, sizeof(name_buf), "l%d_b", i);
        AddXavierFillOp(&init_net, {rows, 1}, name_buf);
    }
    caffe2::CreateNet(init_net, &workspace_)->Run();
}


void Caffe2FeedForwardNetwork::StochasticGradientDescent(
    const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
    size_t mini_batch_size, float learning_rate, const std::vector<Case>* testing_data) {

    caffe2::NetDef predict_net;

    predict_net.set_name("feedforward_predict");
}
