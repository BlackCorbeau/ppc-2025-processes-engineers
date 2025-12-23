#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "remizov_k_systems_linear_equations_gradient/common/include/common.hpp"
#include "remizov_k_systems_linear_equations_gradient/mpi/include/ops_mpi.hpp"
#include "remizov_k_systems_linear_equations_gradient/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace remizov_k_systems_linear_equations_gradient {

class RemizovKSystemLinearEquationsGradientPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kTestSize_ = 10000;

  InType test_input_;
  OutType test_expected_;

  static std::vector<std::vector<double>> CreateSPDMatrix(int n) {
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix[i][j] = 1.0 / (1.0 + std::abs(i - j));
      }
      matrix[i][i] += n;
    }
    return matrix;
  }

  static std::vector<double> CreateVector(int n) {
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) {
      b[i] = static_cast<double>(i + 1);
    }
    return b;
  }

  static bool CheckSolution(const std::vector<double> &result, const std::vector<double> &expected,
                            double tolerance = 1e-2) {
    if (result.size() != expected.size()) {
      return false;
    }

    double max_error = 0.0;
    for (size_t i = 0; i < result.size(); ++i) {
      const double error = std::abs(result[i] - expected[i]);
      max_error = std::max(error, max_error);
    }

    return max_error < tolerance;
  }

  void SetUp() override {
    auto matrix = CreateSPDMatrix(kTestSize_);
    auto vector_b = CreateVector(kTestSize_);
    test_input_ = std::make_tuple(matrix, vector_b);

    test_expected_ = std::vector<double>(kTestSize_, 0.5);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    const auto &[matrix, vector_b] = GetCurrentTestInput();
    return output_data.size() == vector_b.size();
  }

  InType GetTestInputData() final {
    return test_input_;
  }

  InType GetCurrentTestInput() {
    return test_input_;
  }
};

TEST_P(RemizovKSystemLinearEquationsGradientPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RemizovKSystemLinearEquationsGradientMPI,
                                                       RemizovKSystemLinearEquationsGradientSEQ>(
    PPC_SETTINGS_remizov_k_systems_linear_equations_gradient);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RemizovKSystemLinearEquationsGradientPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RemizovKSystemLinearEquationsGradientPerfTest, kGtestValues, kPerfTestName);

}  // namespace remizov_k_systems_linear_equations_gradient
