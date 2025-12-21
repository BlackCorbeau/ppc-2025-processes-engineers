#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"
#include "remizov_k_banded_horizontal_scheme/mpi/include/ops_mpi.hpp"
#include "remizov_k_banded_horizontal_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace remizov_k_banded_horizontal_scheme {

class RemizovKRunBandedHorizontalScheme : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &test_name = std::get<0>(test_param);
    return test_name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    expected_output_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i].size() != expected_output_[i].size()) {
        return false;
      }
      for (size_t j = 0; j < output_data[i].size(); ++j) {
        if (output_data[i][j] != expected_output_[i][j]) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(RemizovKRunBandedHorizontalScheme, MatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(
        "basic_2x2_mult",
        std::make_tuple(std::vector<std::vector<int>>{{1, 2}, {3, 4}}, std::vector<std::vector<int>>{{5, 6}, {7, 8}}),
        std::vector<std::vector<int>>{{19, 22}, {43, 50}}),

    std::make_tuple("identity_mult",
                    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}},
                                    std::vector<std::vector<int>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}),
                    std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}}),

    std::make_tuple("rectangular_mult",
                    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}},
                                    std::vector<std::vector<int>>{{7, 8}, {9, 10}, {11, 12}}),
                    std::vector<std::vector<int>>{{58, 64}, {139, 154}}),

    std::make_tuple("single_element",
                    std::make_tuple(std::vector<std::vector<int>>{{5}}, std::vector<std::vector<int>>{{3}}),
                    std::vector<std::vector<int>>{{15}}),

    std::make_tuple("negative_numbers",
                    std::make_tuple(std::vector<std::vector<int>>{{-1, 2}, {3, -4}},
                                    std::vector<std::vector<int>>{{5, -6}, {-7, 8}}),
                    std::vector<std::vector<int>>{{-19, 22}, {43, -50}}),

    std::make_tuple("large_numbers",
                    std::make_tuple(std::vector<std::vector<int>>{{100, 200}, {300, 400}},
                                    std::vector<std::vector<int>>{{5, 6}, {7, 8}}),
                    std::vector<std::vector<int>>{{1900, 2200}, {4300, 5000}})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<RemizovKBandedHorizontalSchemeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_remizov_k_banded_horizontal_scheme),
                                           ppc::util::AddFuncTask<RemizovKBandedHorizontalSchemeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_remizov_k_banded_horizontal_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RemizovKRunBandedHorizontalScheme::PrintFuncTestName<RemizovKRunBandedHorizontalScheme>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, RemizovKRunBandedHorizontalScheme, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace remizov_k_banded_horizontal_scheme
