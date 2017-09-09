#ifndef DEEP_LEARNING_LINEAR_ALGEBRA_HPP_
#define DEEP_LEARNING_LINEAR_ALGEBRA_HPP_

#include <algorithm>

#ifdef USE_EIGEN

#include <Eigen/Dense>
#define Vector Eigen::RowVectorXf
#define Matrix Eigen::MatrixXf

#define MAT_SIZE(mat) mat.size()
#define MAT_ROWS(mat) mat.rows()
#define MAT_COLS(mat) mat.cols()
#define MAT_DATA(mat) mat.data()
#define MAX_VAL(mat) mat.maxCoeff()
#define MAX_VAL_INDEX(mat, val) mat.maxCoeff(&val)
#define MAT_T(mat) mat.transpose()
#define MAT_CWISE_MUL(m1, m2) m1 = m1.cwiseProduct(m2);
#define MAT_COL_SUM(mat) mat.colwise().sum()

inline void Zeros(Eigen::RowVectorXf& vec, const Eigen::RowVectorXf& tpl) {
    vec.resize(tpl.size());
    vec.fill(0.f);
}

inline void Zeros(Eigen::MatrixXf& mat, const Eigen::MatrixXf& tpl) {
    mat.resize(tpl.rows(), tpl.cols());
    mat.fill(0.f);
}

inline void Ones(Eigen::RowVectorXf& vec, const Eigen::RowVectorXf& tpl) {
    vec.resize(tpl.size());
    vec.fill(1.f);
}

inline void Ones(Eigen::MatrixXf& mat, const Eigen::MatrixXf& tpl) {
    mat.resize(tpl.rows(), tpl.cols());
    mat.fill(1.f);
}

inline void Randomize(Eigen::RowVectorXf& vec, int size) {
    vec.resize(size);
    vec.setRandom();
}

inline void Randomize(Eigen::MatrixXf& mat, int rows, int cols) {
    mat.resize(rows, cols);
    mat.setRandom();
}

inline void ApplyOnLeft(Eigen::RowVectorXf& vec, const Eigen::MatrixXf& other) {
    vec.applyOnTheLeft(other);
}

inline int Rank(const Eigen::MatrixXf& mat) {
    Eigen::FullPivLU<Eigen::MatrixXf> lu_decomp(mat);
    return lu_decomp.rank();
}

inline Eigen::RowVectorXf ReLU(const Eigen::RowVectorXf& z) {
    return z.cwiseMax(0.f);
}

#elif defined(USE_ARMADILLO)

#include <armadillo>
#define Vector arma::Row<float>
#define Matrix arma::Mat<float>

#define MAT_SIZE(mat) mat.n_elem
#define MAT_ROWS(mat) mat.n_rows
#define MAT_COLS(mat) mat.n_cols
#define MAT_DATA(mat) mat.memptr()
#define MAX_VAL(mat) mat.max()
#define MAX_VAL_INDEX(mat, val) val = mat.index_max()
#define MAT_T(mat) mat.t()
#define MAT_CWISE_MUL(m1, m2) m1 %= m2
#define MAT_COL_SUM(mat) arma::sum(mat, 0);

inline void Zeros(arma::Row<float>& vec, const arma::Row<float>& tpl) {
    vec.zeros(tpl.n_elem);
}

inline void Zeros(arma::Mat<float>& mat, const arma::Mat<float>& tpl) {
    mat.zeros(tpl.n_rows, tpl.n_cols);
}

inline void Ones(arma::Row<float>& vec, const arma::Row<float>& tpl) {
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

inline void ApplyOnLeft(arma::Row<float>& vec, const arma::Mat<float>& other) {
    vec = other * vec;
}

inline int Rank(const arma::Mat<float>& mat) {
    return arma::rank(mat);
}

inline arma::Mat<float> ReLU(const arma::Mat<float>& z) {
    arma::Mat<float> a(z.n_rows, z.n_cols);
    for (int i = 0; i < z.n_elem; i++) a(i) = std::max(z(i), 0.f);
    return a;
}

#else

#error "Please define either USE_EIGEN or USE_ARMADILLO"

#endif

inline Vector ReLUDerivative(const Vector& z) {
    Matrix d(MAT_ROWS(z), MAT_COLS(z));
    for (int i = 0; i < MAT_SIZE(z); i++) d(i) = z(i) < 0 ? 0.f : 1.f;
    return d;
}

inline Matrix Sigmoid(const Matrix& z) {
    Matrix a(MAT_ROWS(z), MAT_COLS(z));
    for (int i = 0; i < MAT_SIZE(z); i++) a(i) = 1.f / (1.f + expf(-z(i)));
    return a;
}

inline Matrix SigmoidDerivative(const Matrix& z) {
    const Matrix s = Sigmoid(z);
    Matrix t;
    Ones(t, z);
    t -= s;
    return MAT_CWISE_MUL(t, s);
}

inline Matrix SoftMax(const Matrix& z) {
    Matrix a(MAT_ROWS(z), MAT_COLS(z));
    for (int i = 0; i < MAT_ROWS(z); i++) {
        const auto& z_row = z.row(i);
        const float z_max = MAX_VAL(z_row);
        auto row = a.row(i);
        float sum = 0.f;
        for (int j = 0; j < MAT_COLS(z); j++) {
            const float e = expf(z_row(j) - z_max);
            row(j) = e;
            sum += e;
        }
        row /= sum;
    }
    return a;
}

#endif  // DEEP_LEARNING_LINEAR_ALGEBRA_HPP_
