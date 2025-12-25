#include "remizov_k_systems_linear_equations_gradient/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <vector>

#include "remizov_k_systems_linear_equations_gradient/common/include/common.hpp"

namespace remizov_k_systems_linear_equations_gradient {

RemizovKSystemLinearEquationsGradientSEQ::RemizovKSystemLinearEquationsGradientSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKSystemLinearEquationsGradientSEQ::ValidationImpl() {
  const auto &[matrix, vector_b] = GetInput();

  if (matrix.empty() && vector_b.empty()) {
    return true;
  }

  if (matrix.empty() || vector_b.empty()) {
    return false;
  }

  const size_t n = matrix.size();

  if (matrix[0].size() != n) {
    return false;
  }

  if (vector_b.size() != n) {
    return false;
  }

  return true;
}

bool RemizovKSystemLinearEquationsGradientSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

double ComputeDotProduct(const std::vector<double> &a, const std::vector<double> &b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

void ComputeMatrixVectorProduct(const std::vector<std::vector<double>> &matrix, const std::vector<double> &x,
                                std::vector<double> &result) {
  const size_t n = matrix.size();
  std::ranges::fill(result, 0.0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      result[i] += matrix[i][j] * x[j];
    }
  }
}

void UpdateSolutionAndResidual(double alpha, std::vector<double> &x, std::vector<double> &r,
                               const std::vector<double> &p, const std::vector<double> &ap) {
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
  }
}

void UpdateSearchDirection(double beta, std::vector<double> &p, const std::vector<double> &r) {
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = r[i] + (beta * p[i]);
  }
}

}  // namespace

bool RemizovKSystemLinearEquationsGradientSEQ::RunImpl() {
  const auto &[matrix, vector_b] = GetInput();

  if (matrix.empty() && vector_b.empty()) {
    GetOutput() = std::vector<double>();
    return true;
  }

  const size_t n = matrix.size();

  if (n == 0) {
    GetOutput() = std::vector<double>();
    return true;
  }

  const double tolerance = 1e-6;
  const auto max_iterations = static_cast<int>(n * 2);

  std::vector<double> x(n, 0.0);
  std::vector<double> r(n);
  std::vector<double> p(n);
  std::vector<double> ap(n);

  std::ranges::copy(vector_b, r.begin());
  std::ranges::copy(r, p.begin());

  double r_sq = ComputeDotProduct(r, r);
  const double initial_r_norm = std::sqrt(r_sq);

  if (initial_r_norm < tolerance) {
    GetOutput() = x;
    return true;
  }

  int iteration = 0;

  while (iteration < max_iterations) {
    ComputeMatrixVectorProduct(matrix, p, ap);

    double p_ap = ComputeDotProduct(p, ap);

    if (std::abs(p_ap) < 1e-15) {
      p_ap = r_sq;
    }

    const double alpha = r_sq / p_ap;

    UpdateSolutionAndResidual(alpha, x, r, p, ap);

    double r_sq_new = ComputeDotProduct(r, r);

    if (std::sqrt(r_sq_new) < tolerance) {
      break;
    }

    const double beta = r_sq_new / r_sq;

    UpdateSearchDirection(beta, p, r);

    r_sq = r_sq_new;
    iteration++;
  }

  GetOutput() = x;
  return true;
}

bool RemizovKSystemLinearEquationsGradientSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_systems_linear_equations_gradient
