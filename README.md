# Deep Learning Practices
=========================

## Table of contents
  * [BLAS Performance](#blas-performance)

## BLAS Performance
Based on Armadillo-7.950.1 running on Linux i5-5575R 4.8.0-58-generic.

| BLAS | Matrix Multiplication | Matrix Rank | Simple Network |
| ---- | --------------------- | ----------- | -------------- |
| ATLAS-3.10.3 | x1.0 | x1.0 | x1.0* |
| OpenBLAS-0.2.19 | ~x1.7 | ~x2.0 | ~x2.2 |
| MKL-2017.3.196 | ~x2.0 | ~x2.2 | ~x2.6 |

See cc/benchmark.cc for details.
* On small network(till 512 neurons), ATLAS is actually comparable to MKL.
