#ifndef DEEP_LEARNING_NETWORK_HPP_
#define DEEP_LEARNING_NETWORK_HPP_

#include <inttypes.h>

#include <utility>
#include <vector>

#include <armadillo>

class Network {
  public:
    explicit Network(const std::vector<size_t> layer_sizes);

    void StochasticGradientDescent(
        const std::vector<std::pair<arma::Col<float>, int>>& training_data,
        size_t num_samples_per_epoch, size_t epochs, size_t mini_batch_size, float learning_rate,
        const std::vector<std::pair<arma::Col<float>, int>>* testing_data);

    arma::Col<float> FeedForward(const arma::Col<float>& x) const;

    size_t Evaluate(const std::vector<std::pair<arma::Col<float>, int>>& testing_data);

  private:
    void UpdateMiniBatch(
        const std::vector<std::pair<arma::Col<float>, int>>& training_data,
        std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end,
        float learning_rate);

    void BackPropagate(const arma::Col<float>& x, int y,
                       std::vector<arma::Col<float>>& biases_delta,
                       std::vector<arma::Mat<float>>& weights_delta);

    const size_t num_layers_;
    std::vector<arma::Col<float>> biases_;
    std::vector<arma::Mat<float>> weights_;
};

#endif  // DEEP_LEARNING_NETWORK_HPP_
