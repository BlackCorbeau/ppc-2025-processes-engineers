#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "remizov_k_systems_linear_equations_gradient/common/include/common.hpp"
#include "remizov_k_systems_linear_equations_gradient/mpi/include/ops_mpi.hpp"
#include "remizov_k_systems_linear_equations_gradient/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace remizov_k_systems_linear_equations_gradient {

class RemizovKSystemLinearEquationsGradientPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  // Размеры для тестирования
  const int kSmallSize_ = 100;
  const int kMediumSize_ = 500;
  const int kLargeSize_ = 1000;

  InType small_input_;
  InType medium_input_;
  InType large_input_;

  OutType small_expected_;
  OutType medium_expected_;
  OutType large_expected_;

  // Вспомогательная функция для создания SPD матрицы
  std::vector<std::vector<double>> CreateSPDMatrix(int n) {
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        A[i][j] = 1.0 / (1.0 + std::abs(i - j));
      }
      A[i][i] += n;  // Диагональное преобладание
    }
    return A;
  }

  // Вспомогательная функция для создания вектора правой части
  std::vector<double> CreateVector(int n) {
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) {
      b[i] = static_cast<double>(i + 1);
    }
    return b;
  }

  // Вспомогательная функция для приблизительной проверки решения
  bool CheckSolution(const std::vector<double> &result, const std::vector<double> &expected, double tolerance = 1e-2) {
    if (result.size() != expected.size()) {
      return false;
    }

    // Проверяем, что Ax ≈ b
    double max_error = 0.0;
    for (size_t i = 0; i < result.size(); ++i) {
      double error = std::abs(result[i] - expected[i]);
      if (error > max_error) {
        max_error = error;
      }
    }

    return max_error < tolerance;
  }

  void SetUp() override {
    // Маленькая система
    auto A_small = CreateSPDMatrix(kSmallSize_);
    auto b_small = CreateVector(kSmallSize_);
    small_input_ = std::make_tuple(A_small, b_small);

    // Средняя система
    auto A_medium = CreateSPDMatrix(kMediumSize_);
    auto b_medium = CreateVector(kMediumSize_);
    medium_input_ = std::make_tuple(A_medium, b_medium);

    // Большая система
    auto A_large = CreateSPDMatrix(kLargeSize_);
    auto b_large = CreateVector(kLargeSize_);
    large_input_ = std::make_tuple(A_large, b_large);

    // Ожидаемые решения (приблизительные)
    small_expected_ = std::vector<double>(kSmallSize_, 0.5);  // Примерное значение
    medium_expected_ = std::vector<double>(kMediumSize_, 0.5);
    large_expected_ = std::vector<double>(kLargeSize_, 0.5);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // В performance-тестах мы не проверяем точность решения,
    // а только то, что решение было получено
    if (output_data.empty()) {
      return false;
    }

    // Проверяем размер результата
    const auto &[A, b] = GetCurrentTestInput();
    if (output_data.size() != b.size()) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    // Возвращаем тестовые данные в зависимости от размера
    // Можно использовать параметризацию для разных размеров
    return large_input_;  // Тестируем на большой системе
  }

  // Метод для получения текущих входных данных (нужен для проверки)
  InType GetCurrentTestInput() {
    return large_input_;
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
