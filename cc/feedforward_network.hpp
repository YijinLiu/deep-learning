#ifndef DEEP_LEARNING_FEEDFORWARD_NETWORK_HPP_
#define DEEP_LEARNING_FEEDFORWARD_NETWORK_HPP_

#include <inttypes.h>

#include <utility>
#include <vector>

#include "linear_algebra.hpp"

class FeedForwardNetwork {
  public:
    enum class ActivationFunc {
        Identity,
        ReLU,
        Sigmoid,
        SoftMax
    };

    struct Layer {
        Layer(int num_neurons, ActivationFunc activation) : num_neurons(num_neurons),
                                                            activation(activation) {}
        int num_neurons;
        ActivationFunc activation;
    };

    typedef std::pair<Vector, int> Case;

    FeedForwardNetwork(const std::vector<Layer>& layers, float weight_decay);

    void StochasticGradientDescent(
        const std::vector<Case>& training_data, size_t num_samples_per_epoch, size_t epochs,
        size_t mini_batch_size, float learning_rate, const std::vector<Case>* testing_data);

    size_t Evaluate(const std::vector<Case>& testing_data);

  private:
    void UpdateMiniBatch(
        const std::vector<Case>& training_data,
        std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end,
        float learning_rate);

    Vector Activation(int layer, const Vector& z) const;
    Vector ActivationDerivative(int layer, const Vector& z) const;

    Vector FeedForward(const Vector& x) const;

    void BackPropagate(const Vector& x, int y,
                       std::vector<Vector>& biases_delta, std::vector<Matrix>& weights_delta);

    const std::vector<Layer> layers_;
    const float weight_decay_;
    std::vector<Vector> biases_;
    std::vector<Matrix> weights_;
};

#endif  // DEEP_LEARNING_FEEDFORWARD_NETWORK_HPP_
