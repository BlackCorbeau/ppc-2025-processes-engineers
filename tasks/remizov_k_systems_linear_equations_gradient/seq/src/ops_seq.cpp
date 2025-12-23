#include "remizov_k_systems_linear_equations_gradient/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "remizov_k_systems_linear_equations_gradient/common/include/common.hpp"

namespace remizov_k_systems_linear_equations_gradient {

namespace {

const double kDefaultTolerance = 1e-10;
const int kDefaultMaxIterations = 1000;

}  // namespace

RemizovKSystemLinearEquationsGradientSEQ::RemizovKSystemLinearEquationsGradientSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKSystemLinearEquationsGradientSEQ::ValidationImpl() {
  const auto &[A, b] = GetInput();

  if (A.empty() && b.empty()) {
    return true;
  }

  if (A.empty() || b.empty()) {
    return false;
  }

  size_t n = A.size();

  if (A[0].size() != n) {
    return false;
  }

  if (b.size() != n) {
    return false;
  }

  return true;
}

bool RemizovKSystemLinearEquationsGradientSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RemizovKSystemLinearEquationsGradientSEQ::RunImpl() {
  const auto &[A, b] = GetInput();

  if (A.empty() && b.empty()) {
    GetOutput() = std::vector<double>();
    return true;
  }

  size_t n = A.size();

  if (n == 0) {
    GetOutput() = std::vector<double>();
    return true;
  }

  const double TOLERANCE = 1e-6;
  const int MAX_ITERATIONS = n * 2;

  std::vector<double> x(n, 0.0);
  std::vector<double> r(n);
  std::vector<double> p(n);
  std::vector<double> Ap(n);

  std::copy(b.begin(), b.end(), r.begin());

  std::copy(r.begin(), r.end(), p.begin());

  double r_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    r_sq += r[i] * r[i];
  }
  double initial_r_norm = std::sqrt(r_sq);

  if (initial_r_norm < TOLERANCE) {
    GetOutput() = x;
    return true;
  }

  int iteration = 0;

  while (iteration < MAX_ITERATIONS) {
    std::fill(Ap.begin(), Ap.end(), 0.0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        Ap[i] += A[i][j] * p[j];
      }
    }

    double pAp = 0.0;
    for (size_t i = 0; i < n; ++i) {
      pAp += p[i] * Ap[i];
    }

    if (std::abs(pAp) < 1e-15) {
      pAp = r_sq;
    }

    double alpha = r_sq / pAp;

    for (size_t i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double r_sq_new = 0.0;
    for (size_t i = 0; i < n; ++i) {
      r_sq_new += r[i] * r[i];
    }

    if (std::sqrt(r_sq_new) < TOLERANCE) {
      break;
    }

    double beta = r_sq_new / r_sq;

    for (size_t i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }

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
