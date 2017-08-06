# Deep Learning Practices
=========================

## Table of contents
  * [Performance](#blas-performance)
    * [BLAS](#blas)
    * [Python VS C++](#python-vs-c)

## Performance

### BLAS
Based on Armadillo-7.950.1 running on Linux i5-5575R 4.8.0-58-generic.

| BLAS | Matrix Multiplication | Matrix Rank | Simple Network |
| ---- | --------------------- | ----------- | -------------- |
| ATLAS-3.10.3 | x1.0 | x1.0 | x1.0<sup>[1]</sup> |
| OpenBLAS-0.2.19 | ~x1.7 | ~x2.0 | ~x2.2 |
| MKL-2017.3.196 | ~x2.0 | ~x2.2 | ~x2.6 |

See cc/benchmark.cc for details.

[1] On small network(till 512 neurons), ATLAS is actually comparable to MKL.

### Python VS C++

| Language | Matrix Multiplication | Matrix Rank | Simple Network |
| ---- | --------------------- | ----------- | -------------- |
| Python 2.7 | x1.0 | x1.0 | x1.0 |
| C++ | x1.0 | x1.3-2.0 | x2.0-4.0 |

See cc/benchmark.cc and py/benchmark.py for details.
