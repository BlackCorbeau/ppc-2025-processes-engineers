#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"
#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"
#include "remizov_k_max_in_matrix_string/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace remizov_k_max_in_matrix_string {

class RemizovKRunFuncMaxInMatrixString : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto& input_matrix = std::get<0>(test_param);
    const auto& expected_output = std::get<1>(test_param);

    std::string test_name = "matrix_" + std::to_string(input_matrix.size()) +
                           "x" + (input_matrix.empty() ? "0" : std::to_string(input_matrix[0].size()));
    if (!expected_output.empty()) {
      test_name += "_maxes";
      for (size_t i = 0; i < expected_output.size(); ++i) {
        test_name += "_" + FormatNumber(expected_output[i]);
      }
    }

    return test_name;
  }

 private:
  static std::string FormatNumber(int num) {
    if (num >= 0) {
      return std::to_string(num);
    } else {
      return "neg" + std::to_string(-num);
    }
  }

 protected:
  void SetUp() override {
    int width = -1;
    int height = -1;
    int channels = -1;
    std::vector<uint8_t> img;
    // Read image
    //{
    //  std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_max_in_matrix_string, "pic.jpg");
    //  auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 0);
    //  if (data == nullptr) {
    //    throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
    //  }
    //  img = std::vector<uint8_t>(data, data + (static_cast<ptrdiff_t>(width * height * channels)));
    //  stbi_image_free(data);
    //  if (std::cmp_not_equal(width, height)) {
    //    throw std::runtime_error("width != height: ");
    //  }
    //}

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = width - height + std::min(std::accumulate(img.begin(), img.end(), 0), channels);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(RemizovKRunFuncMaxInMatrixString, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RemizovKMaxInMatrixStringMPI, InType>(kTestParam, PPC_SETTINGS_example_processes),
                   ppc::util::AddFuncTask<RemizovKMaxInMatrixStringMPI, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RemizovKRunFuncMaxInMatrixString::PrintFuncTestName<RemizovKRunFuncMaxInMatrixString>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, RemizovKRunFuncMaxInMatrixString, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace remizov_k_max_in_matrix_string
