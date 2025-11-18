#include "remizov_k_max_in_matrix_string/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"

namespace remizov_k_max_in_matrix_string {

RemizovKMaxInMatrixStringSEQ::RemizovKMaxInMatrixStringSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = std::vector<std::vector<int>>(in);
  GetOutput() = std::vector<int>();
}

bool RemizovKMaxInMatrixStringSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  if (!GetOutput().empty()) {
    return false;
  }

  const std::size_t first_row_size = GetInput()[0].size();
  if (first_row_size == 0) {
    return false;
  }

  return std::ranges::all_of(GetInput(), [first_row_size](const auto &row) { return row.size() == first_row_size; });
}

bool RemizovKMaxInMatrixStringSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().reserve(GetInput().size());
  return GetOutput().empty();
}

bool RemizovKMaxInMatrixStringSEQ::RunImpl() {
  for (const auto &row : GetInput()) {
    if (!row.empty()) {
      int max_element = *std::ranges::max_element(row);
      GetOutput().push_back(max_element);
    }
  }

  return !GetOutput().empty();
}

bool RemizovKMaxInMatrixStringSEQ::PostProcessingImpl() {
  if (GetOutput().size() != GetInput().size()) {
    return false;
  }

  for (std::size_t i = 0; i < GetOutput().size(); i++) {
    const auto &row = GetInput()[i];
    int found_max = GetOutput()[i];

    bool max_exists = false;
    for (int value : row) {
      if (value == found_max) {
        max_exists = true;
        break;
      }
    }

    if (!max_exists) {
      return false;
    }
    for (int value : row) {
      if (value > found_max) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace remizov_k_max_in_matrix_string
