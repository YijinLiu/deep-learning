#ifndef DEEP_LEARNING_FEEDFORWARD_NETWORK_HPP_
#define DEEP_LEARNING_FEEDFORWARD_NETWORK_HPP_

#include <inttypes.h>

#include <utility>
#include <vector>

#include <Eigen/Dense>

class FeedForwardNetwork {
  public:
    typedef std::pair<Eigen::VectorXf, int> Case;

    explicit FeedForwardNetwork(const std::vector<size_t> layer_sizes);

    void StochasticGradientDescent(
        const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
        size_t mini_batch_size, float learning_rate, const std::vector<Case>* testing_data);

    size_t Evaluate(const std::vector<Case>& testing_data);

  private:
    void UpdateMiniBatch(
        const std::vector<Case>& training_data,
        std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end,
        float learning_rate);

    Eigen::VectorXf FeedForward(const Eigen::VectorXf& x) const;

    void BackPropagate(const Eigen::VectorXf& x, int y,
                       std::vector<Eigen::VectorXf>& biases_delta,
                       std::vector<Eigen::MatrixXf>& weights_delta);

    const size_t num_layers_;
    std::vector<Eigen::VectorXf> biases_;
    std::vector<Eigen::MatrixXf> weights_;
};

#endif  // DEEP_LEARNING_FEEDFORWARD_NETWORK_HPP_
