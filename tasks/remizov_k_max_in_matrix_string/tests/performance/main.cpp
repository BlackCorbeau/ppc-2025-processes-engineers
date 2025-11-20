#include <gtest/gtest.h>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"
#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"
#include "remizov_k_max_in_matrix_string/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace remizov_k_max_in_matrix_string {

class RemizovKRunPerfMaxInMatrixString : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    const int rows = 1000;
    const int cols = 1000;
    input_data_.resize(rows);

    std::srand(std::time(nullptr));
    for (int i = 0; i < rows; ++i) {
      input_data_[i].resize(cols);
      int max_in_row = 0;
      for (int j = 0; j < cols; ++j) {
        input_data_[i][j] = std::rand() % 10000;
        if (input_data_[i][j] > max_in_row) {
          max_in_row = input_data_[i][j];
        }
      }
      expected_output_.push_back(max_in_row);
    }
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
    ppc::util::MakeAllPerfTasks<InType, RemizovKMaxInMatrixStringMPI, RemizovKMaxInMatrixStringSEQ>(
        PPC_SETTINGS_remizov_k_max_in_matrix_string);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RemizovKRunPerfMaxInMatrixString::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RemizovKRunPerfMaxInMatrixString, kGtestValues, kPerfTestName);

}  // namespace remizov_k_max_in_matrix_string
