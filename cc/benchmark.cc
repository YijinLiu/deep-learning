#include <benchmark/benchmark.h>
#include <caffe2/core/init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe2.hpp"
#include "mnist.hpp"
#include "simple_network.hpp"

namespace {

void BM_MatrixMul(benchmark::State& state) {
    const int size = state.range(0);
    Matrix A;
    Randomize(A, size, size);
    Matrix B;
    Randomize(B, size, size);
    while (state.KeepRunning()) {
        Matrix C = A * B;
    }
}

void BM_MatrixRank(benchmark::State& state) {
    const int size = state.range(0);
    Matrix A;
    Randomize(A, size, size);
    while (state.KeepRunning()) {
        int rank = Rank(A);
    }
}

void BM_SimpleNetwork(benchmark::State& state) {
    const int neurons = state.range(0);
    const auto training_data = LoadMNISTData(nullptr, "train");
    const size_t image_size = training_data[0].first.size();
    while (state.KeepRunning()) {
        std::vector<Layer> layers;
        layers.emplace_back(image_size, ActivationFunc::Identity);
        layers.emplace_back(neurons, ActivationFunc::Sigmoid);
        layers.emplace_back(10, ActivationFunc::SoftMax);
        SimpleNetwork network(layers, 10);
        network.Train(training_data, 1000, 10, 0.9999, 0.5, nullptr);
    }
}

void BM_Caffe2SimpleNetwork(benchmark::State& state) {
    const int neurons = state.range(0);
    const auto training_data = LoadMNISTData(nullptr, "train");
    const size_t image_size = training_data[0].first.size();
    while (state.KeepRunning()) {
        std::vector<Layer> layers;
        layers.emplace_back(image_size, ActivationFunc::Identity);
        layers.emplace_back(neurons, ActivationFunc::Sigmoid);
        layers.emplace_back(10, ActivationFunc::SoftMax);
        Caffe2SimpleNetwork network(layers, 10, 0.9999, 0.5);
        network.Train(training_data, 1000, 10, nullptr);
    }
}

}  // namespace

BENCHMARK(BM_MatrixMul)->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly()
    ->RangeMultiplier(2)->Range(128, 2048);
BENCHMARK(BM_MatrixRank)->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly()
    ->RangeMultiplier(2)->Range(128, 2048);
BENCHMARK(BM_SimpleNetwork)->Unit(benchmark::kMillisecond)->Repetitions(10)
    ->ReportAggregatesOnly()->RangeMultiplier(2)->Range(128, 2048);
BENCHMARK(BM_Caffe2SimpleNetwork)->Unit(benchmark::kMillisecond)->Repetitions(10)
    ->ReportAggregatesOnly()->RangeMultiplier(2)->Range(128, 2048);

int main(int argc, char** argv) {
    gflags::SetCommandLineOption("v", "-1");
    benchmark::Initialize(&argc, argv);
    caffe2::GlobalInit(&argc, &argv);
    ::benchmark::RunSpecifiedBenchmarks();
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
// BM_SimpleNetwork/128             439 ms
// BM_SimpleNetwork/256             790 ms
// BM_SimpleNetwork/512            2180 ms
// BM_SimpleNetwork/1024          12807 ms
// BM_SimpleNetwork/2048          60815 ms
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
// BM_SimpleNetwork/128             506 ms
// BM_SimpleNetwork/256             951 ms
// BM_SimpleNetwork/512            2571 ms
// BM_SimpleNetwork/1024           6881 ms
// BM_SimpleNetwork/2048          24007 ms
//
// 3. Linux i5-5575R 4.8.0-58-generic Armadillo-7.950.1 with MKL-2017.3.196
// BM_MatrixMul/128                  49 us
// BM_MatrixMul/256                 370 us
// BM_MatrixMul/512                2929 us
// BM_MatrixMul/1024              23066 us
// BM_MatrixMul/2048             184564 us
//
// BM_MatrixRank/128                779 us
// BM_MatrixRank/256               3624 us
// BM_MatrixRank/512              22233 us
// BM_MatrixRank/1024            150790 us
// BM_MatrixRank/2048           1591740 us
//
// BM_SimpleNetwork/128             256 ms
// BM_SimpleNetwork/256             471 ms
// BM_SimpleNetwork/512             945 ms
// BM_SimpleNetwork/1024           1933 ms
// BM_SimpleNetwork/2048           3955 ms
//
// 4. Linux i5-5575R 4.8.0-58-generic Eigen-3.3.4 with MKL-2017.3.196
// BM_MatrixMul/128                  52 us
// BM_MatrixMul/256                 379 us
// BM_MatrixMul/512                2965 us
// BM_MatrixMul/1024              23091 us
// BM_MatrixMul/2048             180069 us
//
// BM_MatrixRank/128                772 us
// BM_MatrixRank/256               5587 us
// BM_MatrixRank/512              41647 us
// BM_MatrixRank/1024            332819 us
// BM_MatrixRank/2048           2795885 us
//
// BM_SimpleNetwork/128             207 ms
// BM_SimpleNetwork/256             374 ms
// BM_SimpleNetwork/512             767 ms
// BM_SimpleNetwork/1024           1675 ms
// BM_SimpleNetwork/2048           3582 ms
//
// BM_Caffe2SimpleNetwork/128       161 ms
// BM_Caffe2SimpleNetwork/256       312 ms
// BM_Caffe2SimpleNetwork/512       613 ms
// BM_Caffe2SimpleNetwork/1024     1359 ms
// BM_Caffe2SimpleNetwork/2048     3145 ms
