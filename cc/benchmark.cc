#include <benchmark/benchmark.h>

#include <armadillo>

namespace {

void BM_MatrixMul(benchmark::State& state) {
    const int size = state.range(0);
    arma::Mat<float> A(size, size, arma::fill::randu);
    arma::Mat<float> B(size, size, arma::fill::randu);
    while (state.KeepRunning()) {
        arma::Mat<float> C = A * B;
    }
}

void BM_MatrixRank(benchmark::State& state) {
    const int size = state.range(0);
    arma::Mat<float> A(size, size, arma::fill::randu);
    while (state.KeepRunning()) {
        int rank = arma::rank(A);
    }
}

}  // namespace

BENCHMARK(BM_MatrixMul)->RangeMultiplier(2)->Range(128, 2048);
BENCHMARK(BM_MatrixRank)->RangeMultiplier(2)->Range(128, 2048);

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}

// 1. Linux i5-5575R 4.8.0-58-generic NumPy-1.13.1 with ATLAS-3.10.3
// BM_MatrixMul/128       297072 ns     297068 ns       2367
// BM_MatrixMul/256       885741 ns     885720 ns        779
// BM_MatrixMul/512      6040530 ns    6039036 ns        113
// BM_MatrixMul/1024    48032997 ns   48027634 ns         14
// BM_MatrixMul/2048   356259422 ns  356245540 ns          2
//
// BM_MatrixRank/128     1248244 ns    1242651 ns        547
// BM_MatrixRank/256     8275527 ns    8275407 ns         83
// BM_MatrixRank/512    55673140 ns   55671379 ns         12
// BM_MatrixRank/1024  422724163 ns  422714866 ns          2
// BM_MatrixRank/2048 3456366837 ns 3453005685 ns          1
//
// 2.
