#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"
#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"
#include "remizov_k_max_in_matrix_string/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace remizov_k_max_in_matrix_string {

class RemizovKRunFuncSystemLinearEquationsGradient : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[input, expected] = test_param;
    const auto &[matrix, vector_b] = input;

    std::string test_name;

    if (matrix.empty()) {
      test_name = "empty_system";
    } else {
      const size_t n = matrix.size();
      test_name = "system_" + std::to_string(n) + "x" + std::to_string(n);

      if (n == 1) {
        test_name += "_scalar";
      } else if (n == 2) {
        if (matrix[0][1] == 0.0 && matrix[1][0] == 0.0) {
          test_name += "_diagonal";
        } else {
          test_name += "_full";
        }
      } else if (n == 3) {
        if (matrix[0][0] == 2.0 && matrix[0][1] == 0.0) {
          test_name += "_diagonal";
        } else if (matrix[0][0] == 4.0 && matrix[0][1] == 1.0) {
          test_name += "_spd";
        } else if (matrix[0][0] == 2.0 && matrix[0][1] == -1.0) {
          test_name += "_tridiag";
        }
      } else if (n == 4) {
        test_name += "_identity";
      }
    }

    return test_name;
  }

 private:
  static bool CheckSolution(const std::vector<double> &x, const std::vector<std::vector<double>> &matrix,
                            const std::vector<double> &vector_b, double tolerance = 1e-4) {
    if (matrix.empty() && x.empty() && vector_b.empty()) {
      return true;
    }

    if (x.size() != vector_b.size() || matrix.size() != vector_b.size()) {
      return false;
    }

    double max_residual = 0.0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      double ax_i = 0.0;
      for (size_t j = 0; j < matrix[i].size(); ++j) {
        ax_i += matrix[i][j] * x[j];
      }
      const double residual = std::abs(vector_b[i] - ax_i);
      max_residual = std::max(residual, max_residual);
    }

    double b_norm = 0.0;
    for (double bi : vector_b) {
      b_norm += bi * bi;
    }
    b_norm = std::sqrt(b_norm);

    if (b_norm < 1e-12) {
      return max_residual < tolerance;
    }

    return max_residual / b_norm < tolerance;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[matrix, vector_b] = input_data_;
    return CheckSolution(output_data, matrix, vector_b, 1e-4);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

std::vector<std::vector<double>> CreateDiagonalMatrix(int n, double value = 1.0) {
  std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i) {
    matrix[i][i] = value;
  }
  return matrix;
}

TEST_P(RemizovKRunFuncSystemLinearEquationsGradient, SolveLinearSystem) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(std::make_tuple(std::vector<std::vector<double>>{{5.0}}, std::vector<double>{10.0}),
                    std::vector<double>{2.0}),

    std::make_tuple(
        std::make_tuple(std::vector<std::vector<double>>{{2.0, 0.0}, {0.0, 3.0}}, std::vector<double>{2.0, 3.0}),
        std::vector<double>{1.0, 1.0}),

    std::make_tuple(std::make_tuple(CreateDiagonalMatrix(3, 2.0), std::vector<double>{2.0, 4.0, 6.0}),
                    std::vector<double>{1.0, 2.0, 3.0}),

    std::make_tuple(std::make_tuple(std::vector<std::vector<double>>{{4.0, 1.0, 1.0}, {1.0, 3.0, 0.0}, {1.0, 0.0, 2.0}},
                                    std::vector<double>{6.0, 4.0, 3.0}),
                    std::vector<double>{1.0, 1.0, 1.0}),

    std::make_tuple(std::make_tuple(CreateDiagonalMatrix(4, 1.0), std::vector<double>{1.0, 2.0, 3.0, 4.0}),
                    std::vector<double>{1.0, 2.0, 3.0, 4.0}),

    std::make_tuple(std::make_tuple(std::vector<std::vector<double>>{{100.0, 10.0}, {10.0, 100.0}},
                                    std::vector<double>{110.0, 110.0}),
                    std::vector<double>{1.0, 1.0}),

    std::make_tuple(
        std::make_tuple(std::vector<std::vector<double>>{{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}},
                        std::vector<double>{1.0, 0.0, 1.0}),
        std::vector<double>{0.75, 0.5, 0.75}),

    std::make_tuple(std::make_tuple(std::vector<std::vector<double>>{}, std::vector<double>{}), std::vector<double>{})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RemizovKSystemLinearEquationsGradientMPI, InType>(
                                               kTestParam, PPC_SETTINGS_remizov_k_systems_linear_equations_gradient),
                                           ppc::util::AddFuncTask<RemizovKSystemLinearEquationsGradientSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_remizov_k_systems_linear_equations_gradient));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    RemizovKRunFuncSystemLinearEquationsGradient::PrintFuncTestName<RemizovKRunFuncSystemLinearEquationsGradient>;

INSTANTIATE_TEST_SUITE_P(LinearSystemTests, RemizovKRunFuncSystemLinearEquationsGradient, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace remizov_k_max_in_matrix_string
