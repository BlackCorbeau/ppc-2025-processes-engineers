#include "remizov_k_banded_horizontal_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"

namespace remizov_k_banded_horizontal_scheme {

namespace {

bool AreRowsConsistent(const Matrix &matrix) {
  if (matrix.empty()) {
    return true;
  }

  const size_t first_row_size = matrix[0].size();
  return std::all_of(matrix.begin(), matrix.end(),
                     [first_row_size](const auto &row) { return row.size() == first_row_size; });
}

}  // namespace

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

  if (!AreRowsConsistent(a) || !AreRowsConsistent(b)) {
    return false;
  }

  return a[0].size() == b.size();
}

Matrix RemizovKBandedHorizontalSchemeSEQ::MultiplyMatrices(const Matrix &a, const Matrix &b) {
  const size_t n = a.size();
  const size_t m = a[0].size();
  const size_t p = b[0].size();

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
