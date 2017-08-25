#include <gflags/gflags.h>
#include <glog/logging.h>

#include "feedforward_network.hpp"
#include "mnist.hpp"

DEFINE_int32(neurons, 30, "");
DEFINE_int32(epochs, 30, "");
DEFINE_int32(mini_batch_size, 10, "");
DEFINE_int32(num_samples_per_epoch, 60000, "");
DEFINE_double(learning_rate, 3.0, "");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    VLOG(1) << "Loading MNIST data into memory ...";
    const auto training_data = LoadMNISTData(nullptr, "train");
    const auto testing_data = LoadMNISTData(nullptr, "t10k");
    VLOG(1) << "Training using MNIST data ...";
    const size_t image_size = training_data[0].first.size();
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(image_size);
    layer_sizes.push_back(FLAGS_neurons);
    layer_sizes.push_back(10);
    FeedForwardNetwork network(layer_sizes);
    network.StochasticGradientDescent(training_data, FLAGS_num_samples_per_epoch, FLAGS_epochs,
                                      FLAGS_mini_batch_size, FLAGS_learning_rate, &testing_data);
}
