#include "caffe2.hpp"

#include <stdio.h>

#include <string>
#include <vector>

#include <caffe2/core/operator.h>
#include <caffe2/core/workspace.h>
#include <glog/logging.h>

#define INIT_NET_NAME "init"
#define TRAIN_NET_NAME "train"
#define PREDICT_NET_NAME "predict"

#define INPUTS "inputs"
#define LABELS "labels"
#define COST "cost"
#define GRAD_SUFFIX "_grad"
#define COST_GRAD COST GRAD_SUFFIX
#define ONE "one"
#define LR "lr"
#define ACCURACY "accuracy"

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
    caffe2::Workspace workspace;

    // Init net.
    caffe2::NetDef init_net_def;
    init_net_def.set_name(INIT_NET_NAME);
    {
        auto* op = AddOp(&init_net_def, "ConstantFill", {}, {ONE});
        AddOpArg(op, "shape", {1});
        AddOpArg(op, "value", 1.f);
    }
    // TODO: Use LearningRate operator.
    {
        auto* op = AddOp(&init_net_def, "ConstantFill", {}, {LR});
        AddOpArg(op, "shape", {1});
        AddOpArg(op, "value", learning_rate);
    }
    // Init parameters for every layer.
    for (int i = 1; i < layers.size(); i++) {
        const int rows = layers[i].num_neurons;
        const int cols = layers[i-1].num_neurons;
        AddXavierFillOp(&init_net_def, {rows, cols}, LayerW(i));
        AddXavierFillOp(&init_net_def, {rows, 1}, LayerB(i));
    }
    caffe2::CreateNet(init_net_def, &workspace)->Run();

    // Train net.
    caffe2::NetDef train_net;
    train_net_def.set_name(TRAIN_NET_NAME);
    AddLayers(&train_net_def, true);
    auto train_net = caffe2::CreateNet(train_net_def, &workspace);

    // Create test predict net.
    std::unique_ptr<caffe2::NetBase> predict_net;
    if (testing_data != nullptr) {
        caffe2::NetDef predict_net_def;
        predict_net_def.set_name(PREDICT_NET_NAME);
        AddLayers(&predict_net_def, false);
        predict_net = caffe2::CreateNet(predict_net_def, &workspace);
    }

    // Train.
    size_t n = std::min(training_data.size(), num_samples_per_epoch);
    std::vector<int> indices(training_data.size());
    for (int i = 0; i < training_data.size(); i++) indices[i] = i;
    srand48(time(NULL));
    for (int e = 0; e < epochs; e++) {
        // Random shuffle.
        for (size_t i = 0; i < n; i++) {
            const size_t step = lrand48() % (training_data.size() - i);
            if (step > 0) std::swap(indices[i], indices[i + step]);
        }
        auto* tensor_cpu = workspace.CreateBlob(INPUTS)->GetMutable<TensorCPU>();
        tensor_cpu->Resize(mini_batch_size, layers_[0].num_neurons);
        float* data = tensor_cpu->mutable_data<float>();
    }
}

#define ADD_OP(...) { \
    auto* op = AddOp(__VA_ARGS__); \
    if (train) gradient_ops.push_back(op); \
}

std::string Caffe2FeedForwardNetwork::AddLayers(caffe2::NetDef* net, bool train) const {
    net->add_external_input(INPUTS);
    net->add_external_input(LABELS);

    // Add feedforward network layers.
    std::vector<const OperatorDef*> gradient_ops;
    std::vector<std::string> params;
    std::string a = INPUTS;
    for (int i = 1; i < layers_.size() - 1; i++) {
        const std::string w = LayerW(i);
        const std::string b = LayerB(i);
        const std::string z = LayerZ(i);
        if (train) {
            net->add_external_input(w);
            net->add_external_input(b);
            params.push_back(w);
            params.push_back(b);
        }
        ADD_OP(net, "FC", {a, w, b}, {z});
        if (train) gradient_ops.push_back(op);
        std::string new_a = LayerA(i);
        switch (layers_[i].activation) {
            case ActivationFunc::Identity:
                new_a = z;
                break;
            case ActivationFunc::ReLU:
                ADD_OP(net, "Relu", {z}, {new_a});
                break;
            case ActivationFunc::Sigmoid:
                if (train && i == layers_.size() - 1) {
                    ADD_OP(net, "SigmoidCrossEntropyWithLogits", {z, LABELS}, {COST});
                    AddOp(net, "Sigmoid", {z}, {new_a});
                } else {
                    ADD_OP(net, "Sigmoid", {z}, {new_a});
                }
                break;
            case ActivationFunc::SoftMax:
                if (train && i == layers_.size() - 1) {
                    ADD_OP(net, "SoftmaxWithLoss", {z, LABELS}, {new_a, COST});
                } else {
                    ADD_OP(net, "Softmax", {z}, {new_a});
                }
            default:
                LOG(FATAL)
                    << "Unknown activation function: " << static_cast<int>(layers_[i].activation);
        }
        a = new_a;
    }
    AddOp(net, "Accuracy", {a, LABELS}, {ACCURACY});

    if (train) {
        // Add gradient operators.
        auto* grad = AddOp(net, "ConstantFill", {COST}, {COST_GRAD});
        AddOpArg(grad, "value", 1.f);
        grad->set_is_gradient_op(true);
        for (auto iter = gradient_ops.rbegin(); iter != gradient_ops.rend(); iter++) {
            const auto* op = *iter;
            std::vector<caffe2::GradientWrapper> outputs(op->output_size());
            for (auto i = 0; i < output.size(); i++) {
                outputs[i].dense_ = op->output(i) + GRAD_SUFFIX;
            }
            auto* grad = train_net.add_op();
            GradientOpsMeta meta = caffe2::GetGradientForOp(*op, outputs);
            grad->CopyFrom(meta.ops_[0]);
            grad->set_is_gradient_op(true);
        }

        // Adjust parameters.
        net->add_external_input(ONE);
        net->add_external_input(LR);
        for (const auto& param : params) {
            AddOp(net, "WeightedSum", {param, ONE, param + GRAD_SUFFIX, LR}, {param});
        }
    }

    return a;
}
