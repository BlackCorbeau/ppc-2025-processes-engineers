#include <gtest/gtest.h>

#include <vector>
#include <iostream>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"
#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"
#include "remizov_k_max_in_matrix_string/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace remizov_k_max_in_matrix_string {

class RemizovKRunPerfMaxInMatrixString : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  OutType expected_output_{};

  void SetUp() override {
    input_data_ = {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9}
    };

    expected_output_ = {3, 6, 9};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RemizovKRunPerfMaxInMatrixString, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RemizovKMaxInMatrixStringMPI, RemizovKMaxInMatrixStringSEQ>(PPC_SETTINGS_remizov_k_max_in_matrix_string);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RemizovKRunPerfMaxInMatrixString::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RemizovKRunPerfMaxInMatrixString, kGtestValues, kPerfTestName);

}  // namespace remizov_k_max_in_matrix_string
