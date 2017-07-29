# Deep Learning Practices
=========================

## Table of contents
  * [BLAS Performance](#blas-performance)

## BLAS Performance
Based on NumPy-1.13.1 running on Linux i5-5575R 4.8.0-58-generic.

| BLAS | Matrix Multiplication | Matrix Rank | Simple Network |
| ---- | --------------------- | ----------- | -------------- |
| ATLAS-3.10.3 | x1.0 | x1.0 | x1.0 |
| OpenBLAS-0.2.19 | ~x1.7 | ~x1.4 | ~x1.02 |
| MKL-2017.3.196 | ~x2.0 | ~x1.7 | ~x1.2 |
