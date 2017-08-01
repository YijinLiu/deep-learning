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

// 1. Linux i5-5575R 4.8.0-58-generic Armadillo-7.950.1 with ATLAS-3.10.3
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
// 2. Linux i5-5575R 4.8.0-58-generic Armadillo-7.950.1 with OpenBLAS-0.2.19
// BM_MatrixMul/128        70116 ns      70115 ns       9573
// BM_MatrixMul/256       458532 ns     458526 ns       1520
// BM_MatrixMul/512      3522217 ns    3521357 ns        199
// BM_MatrixMul/1024    26995594 ns   26995217 ns         26
// BM_MatrixMul/2048   214196776 ns  214195269 ns          3
//
// BM_MatrixRank/128     1281774 ns    1276120 ns        545
// BM_MatrixRank/256     5341550 ns    5341473 ns        125
// BM_MatrixRank/512    29036400 ns   29036090 ns         24
// BM_MatrixRank/1024  178252975 ns  178251867 ns          4
// BM_MatrixRank/2048 1654905208 ns 1651834367 ns          1
//
// 3. Linux i5-5575R 4.8.0-58-generic Armadillo-7.950.1 with MKL-2017.3.196
// BM_MatrixMul/128        14348 ns      14234 ns      50117
// BM_MatrixMul/256       106776 ns     106726 ns       6314
// BM_MatrixMul/512       874009 ns     867167 ns        817
// BM_MatrixMul/1024     6764864 ns    6762785 ns        104
// BM_MatrixMul/2048    52511307 ns   52171988 ns         12
//
// BM_MatrixRank/128      826153 ns     824132 ns        819
// BM_MatrixRank/256     3630703 ns    3615387 ns        192
// BM_MatrixRank/512    14279087 ns   14278752 ns         49
// BM_MatrixRank/1024   73896350 ns   73894769 ns          9
// BM_MatrixRank/2048  739607749 ns  739026617 ns          1
