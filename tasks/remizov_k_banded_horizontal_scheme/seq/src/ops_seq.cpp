#include "remizov_k_banded_horizontal_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
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
  const auto &[A, B] = GetInput();
  return AreMatricesCompatible(A, B);
}

bool RemizovKBandedHorizontalSchemeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::RunImpl() {
  const auto &[A, B] = GetInput();

  if (!AreMatricesCompatible(A, B)) {
    throw std::invalid_argument("Matrices are not compatible for multiplication");
  }

  Matrix result = MultiplyMatrices(A, B);

  GetOutput() = result;
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::PostProcessingImpl() {
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::AreMatricesCompatible(const Matrix &A, const Matrix &B) {
  if (A.empty() || B.empty()) {
    return false;
  }

  size_t cols_A = A[0].size();
  for (const auto &row : A) {
    if (row.size() != cols_A) {
      return false;
    }
  }

  size_t cols_B = B[0].size();
  for (const auto &row : B) {
    if (row.size() != cols_B) {
      return false;
    }
  }

  return A[0].size() == B.size();
}

Matrix RemizovKBandedHorizontalSchemeSEQ::MultiplyMatrices(const Matrix &A, const Matrix &B) {
  size_t n = A.size();
  size_t m = A[0].size();
  size_t p = B[0].size();

  Matrix C(n, std::vector<int>(p, 0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      int sum = 0;
      for (size_t k = 0; k < m; ++k) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }

  return C;
}

}  // namespace remizov_k_banded_horizontal_scheme
