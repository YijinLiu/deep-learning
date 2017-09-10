#include "tensorflow.hpp"

#include <stdlib.h>

#include <iomanip>
#include <limits>

#include <glog/logging.h>
#include <tensorflow/cc/framework/gradients.h>

#define INPUTS "inputs"
#define LABELS "labels"
#define LOSS "loss"
#define GRAD_SUFFIX "_grad"

TensorflowSimpleNetwork::TensorflowSimpleNetwork(
    const std::vector<Layer>& layers, int mini_batch_size)
    : layers_(layers), mini_batch_size_(mini_batch_size),
      scope_(tf::Scope::NewRootScope()), session_(scope_),
      inputs_(scope_.WithOpName(INPUTS), tf::DT_FLOAT,
              tf::ops::Placeholder::Shape({mini_batch_size_, layers[0].num_neurons})),
      labels_(scope_.WithOpName(LABELS), tf::DT_FLOAT,
              tf::ops::Placeholder::Shape({mini_batch_size_})) {
    // Check layers.
    CHECK_GE(layers.size(), 2);

    const auto& input_layer = layers.front();
    if (input_layer.activation != ActivationFunc::Identity) {
        LOG(FATAL)
            << "Input layer's activation function needs to be identity, found "
            << static_cast<int>(input_layer.activation);
    }
    input_size_ = input_layer.num_neurons;

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
    output_classes_ = output_layer.num_neurons;
}

void TensorflowSimpleNetwork::Train(
    const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
    float weight_decay, float learning_rate, const std::vector<Case>* testing_data) {
    // Add layers.
    std::vector<tf::Output> inits;
    std::vector<tf::Output> params;
    tf::Output loss;
    tf::Output a = inputs_;
    for (int i = 1; i < layers_.size(); i++) {
        const int rows = layers_[i].num_neurons;
        const int cols = layers_[i-1].num_neurons;
        auto weight = tf::ops::Variable(scope_.WithOpName(LayerW(i)), {rows, cols}, tf::DT_FLOAT);
        params.push_back(weight);
        inits.push_back(tf::ops::Assign(scope_, weight, tf::ops::ParameterizedTruncatedNormal(
            scope_, tf::ops::Const(scope_, {rows, cols}),
            tf::ops::Const(scope_, {0.f}), tf::ops::Const(scope_, 2.f / (rows + cols)),
            tf::ops::Const(scope_, {std::numeric_limits<float>::infinity()}),
            tf::ops::Const(scope_, {-std::numeric_limits<float>::infinity()}))));
        auto bias = tf::ops::Variable(scope_.WithOpName(LayerB(i)), {rows}, tf::DT_FLOAT);
        params.push_back(bias);
        inits.push_back(tf::ops::Assign(scope_, bias, tf::ops::RandomNormal(
            scope_, tf::ops::Const(scope_, {rows}), tf::DT_FLOAT)));
        auto z = tf::ops::Add(
            scope_.WithOpName(LayerZ(i)),
            tf::ops::MatMul(scope_, weight, a, tf::ops::MatMul::TransposeB(true)), bias);
        tf::Output new_a;
        switch (layers_[i].activation) {
            case ActivationFunc::Identity:
                new_a = z;
                break;
            case ActivationFunc::ReLU:
                new_a = tf::ops::Relu(scope_.WithOpName(LayerA(i)), z);
                break;
            case ActivationFunc::Sigmoid:
                new_a = tf::ops::Sigmoid(scope_.WithOpName(LayerA(i)), z);
                if (i == layers_.size() - 1) {
                    loss = tf::ops::Subtract(scope_.WithOpName(LOSS), new_a, tf::ops::OneHot(
                        scope_, labels_, 10, 1.0f, 0.f));
                }
                break;
            case ActivationFunc::SoftMax:
                new_a = tf::ops::Softmax(scope_.WithOpName(LayerA(i)), z);
                if (i == layers_.size() - 1) {
                    loss = tf::ops::SoftmaxCrossEntropyWithLogits(
                        scope_.WithOpName(LOSS), z, labels_).backprop;
                }
                break;
            default:
                LOG(FATAL)
                    << "Unknown activation function: " << static_cast<int>(layers_[i].activation);
        }
        a = new_a;
    }

    // Compute accuracy.
    accuracy_ = tf::ops::Mean(scope_, tf::ops::Cast(
            scope_, tf::ops::Equal(scope_, tf::ops::ArgMax(scope_, a, 1), labels_), tf::DT_FLOAT), 0);

    // Apply gradients.
    std::vector<tf::Output> param_grads;
    TF_CHECK_OK(tf::AddSymbolicGradients(scope_, {loss}, params, &param_grads));
    std::vector<tf::Output> objectives;
    objectives.push_back(accuracy_);
    auto decay = tf::ops::Const(scope_, {weight_decay});
    auto lr = tf::ops::Const(scope_, {learning_rate});
    for (int i = 0; i < params.size(); i++) {
        objectives.push_back(tf::ops::ApplyGradientDescent(
            scope_.WithOpName(params[i].name() + GRAD_SUFFIX),
            tf::ops::Multiply(scope_, params[i], weight_decay), lr,  param_grads[i]));
    }

    // Init parameters.
    std::vector<tf::Tensor> outputs;
    session_.Run(inits, &outputs);
    
    // Train.
    tf::Tensor batch_inputs(tf::DT_FLOAT, {mini_batch_size_, input_size_});
    float* raw_batch_inputs = batch_inputs.flat<float>().data();
    tf::Tensor batch_labels(tf::DT_INT32, {mini_batch_size_});
    int32_t* raw_batch_labels = batch_labels.flat<int32_t>().data();
    size_t n = std::min(training_data.size(), num_samples_per_epoch);
    std::vector<int> indices(training_data.size());
    for (int i = 0; i < training_data.size(); i++) indices[i] = i;
    srand48(time(NULL));
    for (int e = 0; e < epochs; e++) {
        // Random shuffle.
        for (int i = 0; i < n; i++) {
            const size_t step = lrand48() % (training_data.size() - i);
            if (step > 0) std::swap(indices[i], indices[i + step]);
        }
        float accuracy = 0.0;
        for (int k = 0; k <= n - mini_batch_size_; k += mini_batch_size_) {
            for (int i = 0; i < mini_batch_size_; i++) {
                const auto& c = training_data[indices[k+i]];
                memcpy(raw_batch_inputs + i * input_size_, MAT_DATA(c.first),
                       input_size_ * sizeof(float));
                raw_batch_labels[i] = c.second;
            }
            session_.Run({{inputs_, batch_inputs}, {labels_, batch_labels}}, objectives, &outputs);
            accuracy += outputs[0].scalar<float>()(0);
        }
        accuracy /= (n / mini_batch_size_);
        VLOG(0) << "Epoch " << e + 1 << " training accuracy: " << std::setprecision(4) << accuracy;
        if (testing_data) {
            LOG(INFO) << "Epoch " << e + 1 << " testing accuracy: " << std::setprecision(4)
                << Evaluate(*testing_data);
        }
    }
}

float TensorflowSimpleNetwork::Evaluate(const std::vector<Case>& testing_data) {
    tf::Tensor batch_inputs(tf::DT_FLOAT, {mini_batch_size_, input_size_});
    float* raw_batch_inputs = batch_inputs.flat<float>().data();
    tf::Tensor batch_labels(tf::DT_INT32, {mini_batch_size_});
    int32_t* raw_batch_labels = batch_labels.flat<int32_t>().data();
    std::vector<tf::Output> objectives;
    objectives.push_back(accuracy_);
    std::vector<tf::Tensor> outputs;

    const int n = testing_data.size();
    float accuracy = 0.0;
    for (int k = 0; k <= n - mini_batch_size_; k += mini_batch_size_) {
        for (int i = 0; i < mini_batch_size_; i++) {
            const auto& c = testing_data[k+i];
            memcpy(raw_batch_inputs + i * input_size_, MAT_DATA(c.first),
                   input_size_ * sizeof(float));
            raw_batch_labels[i] = c.second;
        }
        session_.Run({{inputs_, batch_inputs}, {labels_, batch_labels}}, objectives, &outputs);
        accuracy += outputs[0].scalar<float>()(0);
    }
    return accuracy / (n / mini_batch_size_);
}
