#include "remizov_k_banded_horizontal_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"

namespace remizov_k_banded_horizontal_scheme {

RemizovKBandedHorizontalSchemeSEQ::RemizovKBandedHorizontalSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKBandedHorizontalSchemeSEQ::ValidationImpl() {
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::RunImpl() {
  if (GetInput().empty()) {
    return true;
  }

  std::vector<int> result;
  for (const auto &row : GetInput()) {
    if (!row.empty()) {
      int max_val = *std::ranges::max_element(row);
      result.push_back(max_val);
    } else {
      result.push_back(std::numeric_limits<int>::min());
    }
  }

  GetOutput() = result;
  return true;
}

bool RemizovKBandedHorizontalSchemeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_banded_horizontal_scheme
