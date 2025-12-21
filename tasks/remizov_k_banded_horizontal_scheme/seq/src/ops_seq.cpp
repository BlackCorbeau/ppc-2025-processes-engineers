#include "remizov_k_banded_horizontal_scheme/seq/include/ops_seq.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"

namespace remizov_k_banded_horizontal_scheme {

RemizovKBandedHorizontalSchemeSEQ::RemizovKBandedHorizontalSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKBandedHorizontalSchemeSEQ::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return AreMatricesCompatible(a, b);
}

bool RemizovKBandedHorizontalSchemeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::RunImpl() {
  const auto &[a, b] = GetInput();

  if (!AreMatricesCompatible(a, b)) {
    throw std::invalid_argument("Matrices are not compatible for multiplication");
  }

  Matrix result = MultiplyMatrices(a, b);

  GetOutput() = result;
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::PostProcessingImpl() {
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::AreMatricesCompatible(const Matrix &a, const Matrix &b) {
  if (a.empty() || b.empty()) {
    return false;
  }

  size_t cols_a = a[0].size();
  for (const auto &row : a) {
    if (row.size() != cols_a) {
      return false;
    }
  }

  size_t cols_b = b[0].size();
  for (const auto &row : b) {
    if (row.size() != cols_b) {
      return false;
    }
  }

  return a[0].size() == b.size();
}

Matrix RemizovKBandedHorizontalSchemeSEQ::MultiplyMatrices(const Matrix &a, const Matrix &b) {
  size_t n = a.size();
  size_t m = a[0].size();
  size_t p = b[0].size();

  Matrix c(n, std::vector<int>(p, 0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      int sum = 0;
      for (size_t k = 0; k < m; ++k) {
        sum += a[i][k] * b[k][j];
      }
      c[i][j] = sum;
    }
  }

  return c;
}

}  // namespace remizov_k_banded_horizontal_scheme
