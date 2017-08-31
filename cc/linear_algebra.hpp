#ifndef DEEP_LEARNING_LINEAR_ALGEBRA_HPP_
#define DEEP_LEARNING_LINEAR_ALGEBRA_HPP_

#include <algorithm>

#ifdef USE_EIGEN

#include <Eigen/Dense>
#define Vector Eigen::VectorXf
#define Matrix Eigen::MatrixXf

inline int Size(const Eigen::VectorXf& vec) { return vec.size(); }

inline const float* Data(Eigen::VectorXf& vec) { return vec.data(); }

inline float Max(const Eigen::VectorXf& vec) { return vec.maxCoeff(); }

inline int MaxIndex(const Eigen::VectorXf& vec) {
    int index;
    vec.maxCoeff(&index);
    return index;
}

inline void Zeros(Eigen::VectorXf& vec, const Eigen::VectorXf& tpl) {
    vec.resize(tpl.size());
    vec.fill(0.f);
}

inline void Zeros(Eigen::MatrixXf& mat, const Eigen::MatrixXf& tpl) {
    mat.resize(tpl.rows(), tpl.cols());
    mat.fill(0.f);
}

inline void Ones(Eigen::VectorXf& vec, const Eigen::VectorXf& tpl) {
    vec.resize(tpl.size());
    vec.fill(1.f);
}

inline void Ones(Eigen::MatrixXf& mat, const Eigen::MatrixXf& tpl) {
    mat.resize(tpl.rows(), tpl.cols());
    mat.fill(1.f);
}

inline void Randomize(Eigen::VectorXf& vec, int size) {
    vec.resize(size);
    vec.setRandom();
}

inline void Randomize(Eigen::MatrixXf& mat, int rows, int cols) {
    mat.resize(rows, cols);
    mat.setRandom();
}

inline void ElemWiseMul(Eigen::VectorXf& v1, const Eigen::VectorXf& v2) {
    v1 = v1.cwiseProduct(v2);
}

inline Eigen::MatrixXf Transpose(const Eigen::MatrixXf& mat) {
    return mat.transpose();
}

inline void ApplyOnLeft(Eigen::VectorXf& vec, const Eigen::MatrixXf& other) {
    vec.applyOnTheLeft(other);
}

inline int Rank(const Eigen::MatrixXf& mat) {
    Eigen::FullPivLU<Eigen::MatrixXf> lu_decomp(mat);
    return lu_decomp.rank();
}

inline Eigen::VectorXf Sigmoid(const Eigen::VectorXf& z) {
    Eigen::VectorXf s(z.size());
    for (int i = 0; i < z.size(); i++) s(i) = 1.f / (1.f + expf(-z(i)));
    return s;
}

inline Eigen::VectorXf SigmoidDerivative(const Eigen::VectorXf& z) {
    const Eigen::VectorXf s = Sigmoid(z);
    return s.cwiseProduct(Eigen::VectorXf::Constant(z.size(), 1.f) - s);
}

inline Eigen::VectorXf ReLU(const Eigen::VectorXf& z) {
    return z.cwiseMax(0.f);
}

#elif defined(USE_ARMADILLO)

#include <armadillo>
#define Vector arma::Col<float>
#define Matrix arma::Mat<float>

inline int Size(const arma::Col<float>& vec) { return vec.n_elem; }

inline const float* Data(const arma::Col<float>& vec) { return vec.memptr(); }

inline float Max(const arma::Col<float>& vec) { return vec.max(); }

inline int MaxIndex(const arma::Col<float>& vec) { return vec.index_max(); }

inline void Zeros(arma::Col<float>& vec, const arma::Col<float>& tpl) {
    vec.zeros(tpl.n_elem);
}

inline void Zeros(arma::Mat<float>& mat, const arma::Mat<float>& tpl) {
    mat.zeros(tpl.n_rows, tpl.n_cols);
}

inline void Ones(arma::Col<float>& vec, const arma::Col<float>& tpl) {
    vec.ones(tpl.n_elem);
}

inline void Ones(arma::Mat<float>& mat, const arma::Mat<float>& tpl) {
    mat.ones(tpl.n_rows, tpl.n_cols);
}

inline void Randomize(arma::Mat<float>& vec, int size) {
    vec.randu(size);
}

inline void Randomize(arma::Mat<float>& mat, int rows, int cols) {
    mat.randu(rows, cols);
}

inline void ElemWiseMul(arma::Col<float>& v1, const arma::Col<float>& v2) {
    v1 %= v2;
}

inline arma::Mat<float> Transpose(const arma::Mat<float>& mat) {
    return mat.t();
}

inline void ApplyOnLeft(arma::Col<float>& vec, const arma::Mat<float>& other) {
    vec = other * vec;
}

inline int Rank(const arma::Mat<float>& mat) {
    return arma::rank(mat);
}

inline arma::Col<float> Sigmoid(const arma::Col<float>& z) {
    arma::Col<float> s(z.n_elem);
    for (int i = 0; i < z.n_elem; i++) s(i) = 1.f / (1.f + expf(-z[i]));
    return s;
}

inline arma::Col<float> SigmoidDerivative(const arma::Col<float>& z) {
    const arma::Col<float> s = Sigmoid(z);
    return s % (arma::ones<arma::Col<float>>(z.n_elem) - s);
}

inline arma::Col<float> ReLU(const arma::Col<float>& z) {
    arma::Col<float> a(z.n_elem);
    for (int i = 0; i < z.n_elem; i++) a(i) = std::max(z[i], 0.f);
    return a;
}

#else

#error "Please define either USE_EIGEN or USE_ARMADILLO"

#endif

inline Vector ReLUDerivative(const Vector& z) {
    Vector d = Sigmoid(z);
    for (int i = 0; i < Size(z); i++) {
        if (z(i) < 0) {
            d(i) = 0.f;
        } else {
            d(i) = 1.f;
        }
    }
    return d;
}

inline Vector SoftMax(const Vector& z) {
    const int n = Size(z);
    Vector a(n);
    const float max = Max(z);
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
        const float e = expf(z(i) - max);
        a(i) = e;
        sum += e;
    }
    a /= sum;
    return a;
}

#endif  // DEEP_LEARNING_LINEAR_ALGEBRA_HPP_
