#include "caffe2.hpp"

#include <stdio.h>

#include <string>
#include <vector>

#include <caffe2/core/operator.h>
#include <glog/logging.h>

#define INPUTS "inputs"
#define LABELS "labels"
#define COST "cost"
#define GRAD_SUFFIX "_grad"
#define COST_GRAD COST GRAD_SUFFIX
#define ONE "one"
#define LR "lr"

namespace {

std::string Sprintf(const char* fmt, ...) {
    char buf[100];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

std::string LayerW(int l) { return Sprintf("l%d_w", l); }

std::string LayerB(int l) { return Sprintf("l%d_b", l); }

std::string LayerZ(int l) { return Sprintf("l%d_z", l); }

std::string LayerA(int l) { return Sprintf("l%d_a", l); }

caffe2::Argument* AddOpArg(caffe2::OperatorDef* op, const std::string& name) {
    auto* arg = op->add_arg();
    arg->set_name(name);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef* op, const std::string& name, int value) {
    auto* arg = AddOpArg(op, name);
    arg->set_i(value);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef* op, const std::string& name, float value) {
    auto* arg = AddOpArg(op, name);
    arg->set_f(value);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef* op, const std::string& name,
                           const std::string& value) {
    auto* arg = AddOpArg(op, name);
    arg->set_s(value);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef* op, const std::string& name,
                           std::vector<int> values) {
    auto* arg = AddOpArg(op, name);
    for (auto value : values) arg->add_ints(value);
    return arg;
}


caffe2::Argument* AddOpArg(caffe2::OperatorDef* op, const std::string& name,
                           std::vector<float> values) {
    auto* arg = AddOpArg(op, name);
    for (auto value : values) arg->add_floats(value);
    return arg;
}

caffe2::Argument* AddOpArg(caffe2::OperatorDef* op, const std::string& name,
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
    const std::vector<Layer>& layers, float weight_decay) : workspace_("feedforward"),
                                                            layers_(layers),
                                                            weight_decay_(weight_decay) {
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
}

void Caffe2FeedForwardNetwork::StochasticGradientDescent(
    const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
    size_t mini_batch_size, float learning_rate, const std::vector<Case>* testing_data) {
    // Init net.
    caffe2::NetDef init_net;
    init_net.set_name("init");
    {
        auto* op = AddOp(&init_net, "ConstantFill", {}, {ONE});
        AddOpArg(op, "shape", {1});
        AddOpArg(op, "value", 1.f);
    }
    {
        auto* op = AddOp(&init_net, "ConstantFill", {}, {LR});
        AddOpArg(op, "shape", {1});
        AddOpArg(op, "value", learning_rate);
    }
    for (int i = 1; i < layers.size(); i++) {
        const int rows = layers[i].num_neurons;
        const int cols = layers[i-1].num_neurons;
        AddXavierFillOp(&init_net, {rows, cols}, LayerW(i));
        AddXavierFillOp(&init_net, {rows, 1}, LayerB(i));
    }
    caffe2::CreateNet(init_net, &workspace_)->Run();

    // Train net.
    caffe2::NetDef train_net;
    train_net.set_name("train");
    train_net.add_external_input(INPUTS);
    train_net.add_external_input(LABELS);
    train_net.add_external_input(ONE);
    train_net.add_external_input(LR);

    std::vector<const OperatorDef*> gradient_ops;
    std::vector<std::string> params;

    // Add operators for every layer.
    std::string a = INPUTS;
    for (int i = 1; i < layers_.size() - 1; i++) {
        const std::string w = LayerW(i);
        const std::string b = LayerB(i);
        const std::string z = LayerZ(i);
        params.push_back(w);
        params.push_back(b);
        train_net.add_external_input(w);
        train_net.add_external_input(b);
        gradient_ops.push_back(AddOp(&train_net, "FC", {a, w, b}, {z}));
        std::string new_a = LayerA(i);
        switch (layers_[i].activation) {
            case ActivationFunc::Identity:
                new_a = z;
                break;
            case ActivationFunc::ReLU:
                gradient_ops.push_back(AddOp(&train_net, "Relu", {z}, {new_a}));
                break;
            case ActivationFunc::Sigmoid:
                if (i == layers_.size() - 1) {
                    gradient_ops.push_back(
                        AddOp(&train_net, "SigmoidCrossEntropyWithLogits", {z, LABELS}, {COST}));
                } else {
                    gradient_ops.push_back(AddOp(&train_net, "Sigmoid", {z}, {new_a}));
                }
                break;
            case ActivationFunc::SoftMax:
                if (i == layers_.size() - 1) {
                    gradient_ops.push_back(
                        AddOp(&train_net, "SoftmaxWithLoss", {z, LABELS}, {new_a, COST}));
                } else {
                    gradient_ops.push_back(AddOp(&train_net, "Softmax", {z}, {new_a}));
                }
            default:
                LOG(FATAL)
                    << "Unknown activation function: " << static_cast<int>(layers_[I].activation);
        }
        a = new_a;
    }
    AddOp(&train_net, "Iter", {ITER}, {ITER});

    // Add gradient operators.
    {
        auto* grad = AddOp(&train_net, "ConstantFill", {COST}, {COST_GRAD});
        AddOpArg(grad, "value", 1.f);
        grad->set_is_gradient_op(true);
    }
    for (auto iter = gradient_ops.rbegin(); iter != gradient_ops.rend(); iter++) {
        const auto* op = *iter;
        std::vector<caffe2::GradientWrapper> outputs(op->output_size());
        for (auto i = 0; i < output.size(); i++) outputs[i].dense_ = op->output(i) + GRAD_SUFFIX;
        auto* grad = train_net.add_op();
        GradientOpsMeta meta = caffe2::GetGradientForOp(*op, outputs);
        grad->CopyFrom(meta.ops_[0]);
        grad->set_is_gradient_op(true);
    }

    // Adjust parameters.
    for (const auto& param : params) {
        AddOp(&train_net, "WeightedSum", {param, ONE, param + GRAD_SUFFIX, LR}, {param});
    }
}
