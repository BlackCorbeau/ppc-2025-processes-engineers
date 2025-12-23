#include "remizov_k_systems_linear_equations_gradient/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "remizov_k_systems_linear_equations_gradient/common/include/common.hpp"

namespace remizov_k_systems_linear_equations_gradient {

RemizovKSystemLinearEquationsGradientSEQ::RemizovKSystemLinearEquationsGradientSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKSystemLinearEquationsGradientSEQ::ValidationImpl() {
  const auto &[A, b] = GetInput();

  if (A.empty() || b.empty()) {
    return false;
  }

  size_t n = A.size();

  // Проверка квадратности матрицы
  if (A[0].size() != n) {
    return false;
  }

  // Проверка размерности вектора b
  if (b.size() != n) {
    return false;
  }

  // Проверка симметричности и положительной определенности (упрощенно)
  for (size_t i = 0; i < n; ++i) {
    if (A[i][i] <= 0) {
      return false;  // Диагональные элементы должны быть положительными
    }
  }

  return true;
}

bool RemizovKSystemLinearEquationsGradientSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

double RemizovKSystemLinearEquationsGradientSEQ::DotProduct(const std::vector<double> &a,
                                                            const std::vector<double> &b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

std::vector<double> RemizovKSystemLinearEquationsGradientSEQ::MatrixVectorMultiply(
    const std::vector<std::vector<double>> &A, const std::vector<double> &x) {
  size_t n = A.size();
  std::vector<double> result(n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      result[i] += A[i][j] * x[j];
    }
  }

  return result;
}

std::vector<double> RemizovKSystemLinearEquationsGradientSEQ::VectorSubtract(const std::vector<double> &a,
                                                                             const std::vector<double> &b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> RemizovKSystemLinearEquationsGradientSEQ::VectorAdd(const std::vector<double> &a,
                                                                        const std::vector<double> &b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> RemizovKSystemLinearEquationsGradientSEQ::VectorScalarMultiply(const std::vector<double> &v,
                                                                                   double scalar) {
  std::vector<double> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = v[i] * scalar;
  }
  return result;
}

bool RemizovKSystemLinearEquationsGradientSEQ::RunImpl() {
  const auto &[A, b] = GetInput();
  size_t n = A.size();

  if (n == 0) {
    GetOutput() = std::vector<double>();
    return true;
  }

  // Проверка размеров
  if (A[0].size() != n || b.size() != n) {
    GetOutput() = std::vector<double>();
    return false;
  }

  const int MAX_ITERATIONS = static_cast<int>(n * 2);
  const double TOLERANCE = 1e-12;

  // Начальное приближение
  std::vector<double> x(n, 0.0);

  // Начальная невязка r = b - A*x
  std::vector<double> r = b;
  std::vector<double> Ax = MatrixVectorMultiply(A, x);
  for (size_t i = 0; i < n; ++i) {
    r[i] -= Ax[i];
  }

  std::vector<double> p = r;

  double r_norm_squared = DotProduct(r, r);
  double initial_residual = std::sqrt(r_norm_squared);

  if (initial_residual < TOLERANCE) {
    GetOutput() = x;
    return true;
  }

  int iteration = 0;
  double residual_norm = initial_residual;

  while (iteration < MAX_ITERATIONS && residual_norm > TOLERANCE * initial_residual) {
    std::vector<double> Ap = MatrixVectorMultiply(A, p);

    double pAp = DotProduct(p, Ap);

    if (std::abs(pAp) < 1e-15) {
      break;
    }

    double alpha = r_norm_squared / pAp;

    // x = x + alpha * p
    for (size_t i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
    }

    // r = r - alpha * Ap
    for (size_t i = 0; i < n; ++i) {
      r[i] -= alpha * Ap[i];
    }

    double r_norm_squared_old = r_norm_squared;
    r_norm_squared = DotProduct(r, r);
    residual_norm = std::sqrt(r_norm_squared);

    if (std::abs(r_norm_squared_old) < 1e-15) {
      break;
    }

    double beta = r_norm_squared / r_norm_squared_old;

    // p = r + beta * p
    for (size_t i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    iteration++;
  }

  GetOutput() = x;
  return true;
}

bool RemizovKSystemLinearEquationsGradientSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_systems_linear_equations_gradient
