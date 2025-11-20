#include "remizov_k_max_in_matrix_string/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"

namespace remizov_k_max_in_matrix_string {

RemizovKMaxInMatrixStringSEQ::RemizovKMaxInMatrixStringSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKMaxInMatrixStringSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }

  size_t first_row_size = GetInput()[0].size();
  for (const auto &row : GetInput()) {
    if (row.size() != first_row_size) {
      return false;
    }
  }
  return true;
}

bool RemizovKMaxInMatrixStringSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RemizovKMaxInMatrixStringSEQ::RunImpl() {
  GetOutput().reserve(GetInput().size());

  for (const auto &row : GetInput()) {
    if (!row.empty()) {
      GetOutput().push_back(*std::max_element(row.begin(), row.end()));
    }
  }

  return true;
}

bool RemizovKMaxInMatrixStringSEQ::PostProcessingImpl() {
  return GetOutput().size() == GetInput().size();
}

}  // namespace remizov_k_max_in_matrix_string
