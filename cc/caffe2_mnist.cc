#include <caffe2/core/init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe2.hpp"
#include "mnist.hpp"

DEFINE_int32(neurons, 30, "");
DEFINE_int32(epochs, 30, "");
DEFINE_int32(mini_batch_size, 10, "");
DEFINE_int32(num_samples_per_epoch, 60000, "");
DEFINE_double(weight_decay, 0.9999, "");
DEFINE_double(learning_rate, 0.5, "");

int main(int argc, char* argv[]) {
    gflags::SetCommandLineOption("v", "-1");
    caffe2::GlobalInit(&argc, &argv);
    VLOG(-1) << "Loading MNIST data into memory ...";
    const auto training_data = LoadMNISTData(nullptr, "train");
    const auto testing_data = LoadMNISTData(nullptr, "t10k");
    VLOG(-1) << "Training using MNIST data ...";
    const size_t image_size = training_data[0].first.size();
    std::vector<Layer> layers;
    layers.emplace_back(image_size, ActivationFunc::Identity);
    layers.emplace_back(FLAGS_neurons, ActivationFunc::Sigmoid);
    layers.emplace_back(10, ActivationFunc::SoftMax);
    Caffe2FeedForwardNetwork network(
        layers, FLAGS_mini_batch_size, FLAGS_weight_decay, FLAGS_learning_rate);
    network.Train(training_data, FLAGS_num_samples_per_epoch, FLAGS_epochs, &testing_data);
}
