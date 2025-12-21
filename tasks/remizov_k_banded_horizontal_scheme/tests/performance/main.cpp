#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"
#include "remizov_k_banded_horizontal_scheme/mpi/include/ops_mpi.hpp"
#include "remizov_k_banded_horizontal_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace remizov_k_banded_horizontal_scheme {

class RemizovKBandedHorizontalSchemePerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 100;
  InType input_data_;
  OutType res_;

  void SetUp() override {
    Matrix matrix_a(kMatrixSize_, std::vector<int>(kMatrixSize_, 2));
    Matrix matrix_b(kMatrixSize_, std::vector<int>(kMatrixSize_, 3));

    Matrix expected(kMatrixSize_, std::vector<int>(kMatrixSize_, 2 * 3 * kMatrixSize_));

    input_data_ = std::make_tuple(matrix_a, matrix_b);
    res_ = expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != res_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i].size() != res_[i].size()) {
        return false;
      }
      for (size_t j = 0; j < output_data[i].size(); ++j) {
        if (output_data[i][j] != res_[i][j]) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RemizovKBandedHorizontalSchemePerfTest, MatrixMultiplicationPerf) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RemizovKBandedHorizontalSchemeMPI, RemizovKBandedHorizontalSchemeSEQ>(
        PPC_SETTINGS_remizov_k_banded_horizontal_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RemizovKBandedHorizontalSchemePerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationPerf, RemizovKBandedHorizontalSchemePerfTest, kGtestValues, kPerfTestName);

}  // namespace remizov_k_banded_horizontal_scheme
