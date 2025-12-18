#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"
#include "remizov_k_banded_horizontal_scheme/mpi/include/ops_mpi.hpp"
#include "remizov_k_banded_horizontal_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace remizov_k_banded_horizontal_scheme {

class RemizovKBandedHorizontalSchemePerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 50000000;
  InType input_data_;
  OutType res_;

  void SetUp() override {
    int n = static_cast<int>(std::sqrt(kCount_));
    input_data_ = std::vector<std::vector<int>>(n, std::vector<int>(n, 2));
    res_ = std::vector<int>(n, 2);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return res_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RemizovKBandedHorizontalSchemePerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RemizovKBandedHorizontalSchemeMPI, RemizovKBandedHorizontalSchemeSEQ>(
        PPC_SETTINGS_remizov_k_banded_horizontal_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RemizovKBandedHorizontalSchemePerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RemizovKBandedHorizontalSchemePerfTest, kGtestValues, kPerfTestName);

}  // namespace remizov_k_banded_horizontal_scheme
