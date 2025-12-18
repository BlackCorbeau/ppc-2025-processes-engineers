#include "remizov_k_banded_horizontal_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"

namespace remizov_k_banded_horizontal_scheme {

RemizovKBandedHorizontalSchemeMPI::RemizovKBandedHorizontalSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKBandedHorizontalSchemeMPI::ValidationImpl() {
  return true;
}

bool RemizovKBandedHorizontalSchemeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RemizovKBandedHorizontalSchemeMPI::RunImpl() {
  return true;
}

bool RemizovKBandedHorizontalSchemeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_banded_horizontal_scheme
