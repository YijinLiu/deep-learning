#include <benchmark/benchmark.h>

#include "mnist.hpp"
#include "feedforward_network.hpp"

namespace {

void BM_MatrixMul(benchmark::State& state) {
    const int size = state.range(0);
    Eigen::MatrixXf A(size, size);
    A.setRandom();
    Eigen::MatrixXf B(size, size);
    B.setRandom();
    while (state.KeepRunning()) {
        Eigen::MatrixXf C = A * B;
    }
}

void BM_MatrixRank(benchmark::State& state) {
    const int size = state.range(0);
    Eigen::MatrixXf A(size, size);
    A.setRandom();
    while (state.KeepRunning()) {
        Eigen::FullPivLU<Eigen::MatrixXf> lu_decomp(A);
        int rank = lu_decomp.rank();
    }
}

void BM_FeedForwardNetwork(benchmark::State& state) {
    const int neurons = state.range(0);
    const auto training_data = LoadMNISTData(nullptr, "train");
    const size_t image_size = training_data[0].first.size();
    while (state.KeepRunning()) {
        std::vector<size_t> layer_sizes;
        layer_sizes.push_back(image_size);
        layer_sizes.push_back(neurons);
        layer_sizes.push_back(10);
        FeedForwardNetwork network(layer_sizes);
        network.StochasticGradientDescent(training_data, 1000, 10, 10, 3.0, nullptr);
    }
}

}  // namespace

BENCHMARK(BM_MatrixMul)->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly()
    ->RangeMultiplier(2)->Range(128, 2048);
BENCHMARK(BM_MatrixRank)->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly()
    ->RangeMultiplier(2)->Range(128, 2048);
BENCHMARK(BM_FeedForwardNetwork)->Unit(benchmark::kMillisecond)->Repetitions(10)
    ->ReportAggregatesOnly()->RangeMultiplier(2)->Range(128, 2048);

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}

// 1. Linux i5-5575R 4.8.0-58-generic Armadillo-7.950.1 with ATLAS-3.10.3
// BM_MatrixMul/128                 296 us
// BM_MatrixMul/256                 882 us
// BM_MatrixMul/512                6028 us
// BM_MatrixMul/1024              48629 us
// BM_MatrixMul/2048             359077 us
//
// BM_MatrixRank/128               1261 us
// BM_MatrixRank/256               8090 us
// BM_MatrixRank/512              55663 us
// BM_MatrixRank/1024            415310 us
// BM_MatrixRank/2048           3452130 us
//
// BM_FeedForwardNetwork/128        439 ms
// BM_FeedForwardNetwork/256        790 ms
// BM_FeedForwardNetwork/512       2180 ms
// BM_FeedForwardNetwork/1024     12807 ms
// BM_FeedForwardNetwork/2048     60815 ms
//
// 2. Linux i5-5575R 4.8.0-58-generic Armadillo-7.950.1 with OpenBLAS-0.2.19
// BM_MatrixMul/128                  70 us
// BM_MatrixMul/256                 461 us
// BM_MatrixMul/512                3532 us
// BM_MatrixMul/1024              27478 us
// BM_MatrixMul/2048             215039 us
//
// BM_MatrixRank/128               1275 us
// BM_MatrixRank/256               5496 us
// BM_MatrixRank/512              29618 us
// BM_MatrixRank/1024            174341 us
// BM_MatrixRank/2048           1675364 us
//
// BM_FeedForwardNetwork/128        506 ms
// BM_FeedForwardNetwork/256        951 ms
// BM_FeedForwardNetwork/512       2571 ms
// BM_FeedForwardNetwork/1024      6881 ms
// BM_FeedForwardNetwork/2048     24007 ms
//
// 3. Linux i5-5575R 4.8.0-58-generic Armadillo-7.950.1 with MKL-2017.3.196
// BM_MatrixMul/128                  51 us
// BM_MatrixMul/256                 383 us
// BM_MatrixMul/512                3049 us
// BM_MatrixMul/1024              24705 us
// BM_MatrixMul/2048             197104 us
//
// BM_MatrixRank/128                772 us
// BM_MatrixRank/256               3547 us
// BM_MatrixRank/512              22423 us
// BM_MatrixRank/1024            153508 us
// BM_MatrixRank/2048           1592578 us
//
// BM_FeedForwardNetwork/128        446 ms
// BM_FeedForwardNetwork/256        850 ms
// BM_FeedForwardNetwork/512       2142 ms
// BM_FeedForwardNetwork/1024      5839 ms
// BM_FeedForwardNetwork/2048     19566 ms
