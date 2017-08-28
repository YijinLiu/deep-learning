#ifndef DEEP_LEARNING_LINEAR_ALGEBRA_HPP_
#define DEEP_LEARNING_LINEAR_ALGEBRA_HPP_

#ifdef USE_EIGEN

#include <Eigen/Dense>
#define Vector Eigen::VectorXf
#define Matrix Eigen::MatrixXf

inline void InitializeVector(Eigen::VectorXf& vec, const Eigen::VectorXf& tpl) {
    vec.resize(tpl.size());
    vec.fill(0.f);
}

inline void InitializeMatrix(Eigen::MatrixXf& mat, const Eigen::MatrixXf& tpl) {
    mat.resize(tpl.rows(), tpl.cols());
    mat.fill(0.f);
}

inline void RandomizeVector(Eigen::VectorXf& vec, int size) {
    vec.resize(size);
    vec.setRandom();
}

inline void RandomizeMatrix(Eigen::MatrixXf& mat, int rows, int cols) {
    mat.resize(rows, cols);
    mat.setRandom();
}

inline int IndexMax(const Eigen::VectorXf& vec) {
    int index;
    vec.maxCoeff(&index);
    return index;
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

#elif defined(USE_ARMADILLO)

#include <armadillo>
#define Vector arma::Col<float>
#define Matrix arma::Mat<float>

inline void InitializeVector(arma::Col<float>& vec, const arma::Col<float>& tpl) {
    vec.zeros(tpl.n_elem);
}

inline void InitializeMatrix(arma::Mat<float>& mat, const arma::Mat<float>& tpl) {
    mat.zeros(tpl.n_rows, tpl.n_cols);
}

inline void RandomizeVector(arma::Mat<float>& vec, int size) {
    vec.randn(size);
}

inline void RandomizeMatrix(arma::Mat<float>& mat, int rows, int cols) {
    mat.randn(rows, cols);
}

inline int IndexMax(const arma::Col<float>& vec) {
    return vec.index_max();
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
    for (int i = 0; i < z.n_elem; i++) s[i] = 1.f / (1.f + expf(-z[i]));
    return s;
}

inline arma::Col<float> SigmoidDerivative(const arma::Col<float>& z) {
    const arma::Col<float> s = Sigmoid(z);
    return s % (arma::ones<arma::Col<float>>(z.n_elem) - s);
}

#else

#error "Please define either USE_EIGEN or USE_ARMADILLO"

#endif

#endif  // DEEP_LEARNING_LINEAR_ALGEBRA_HPP_
