#include "remizov_k_max_in_matrix_string/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "example_processes/common/include/common.hpp"
#include "util/include/util.hpp"

namespace remizov_k_max_in_matrix_string{

RemizovKMaxInMatrixStringSEQ::RemizovKMaxInMatrixStringSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool RemizovKMaxInMatrixStringSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  if (!GetOutput().empty()) {
    return false;
  }

  size_t first_row_size = GetInput()[0].size();
  if (first_row_size == 0) {
    return false;
  }

  for (const auto& row : GetInput()) {
    if (row.size() != first_row_size) {
      return false;
    }
  }

  return true;
}

bool RemizovKMaxInMatrixStringSEQ::PreProcessingImpl() {
  GetOutput().clear();  // Очищаем выходной вектор
  GetOutput().reserve(GetInput().size());
  return GetOutput().empty();
}

bool RemizovKMaxInMatrixStringSEQ::RunImpl() {
  for (const auto& row : GetInput()) {
     if (!row.empty()) {
        int max_element = *std::max_element(row.begin(), row.end());
        GetOutput().push_back(max_element);
     }
  }

  return !GetOutput().empty();
}

bool RemizovKMaxInMatrixStringSEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace remizov_k_max_in_matrix_string
