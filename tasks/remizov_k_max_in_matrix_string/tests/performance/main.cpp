#include <gtest/gtest.h>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"
#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"
#include "remizov_k_max_in_matrix_string/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace remizov_k_max_in_matrix_string{

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RemizovKMaxInMatrixStringMPI, RemizovKMaxInMatrixStringMPI>(PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace remizov_k_max_in_matrix_string
