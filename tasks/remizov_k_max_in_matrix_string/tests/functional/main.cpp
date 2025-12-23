#include <gtest/gtest.h>

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

class RemizovKRunFuncSystemLinearEquationsGradient
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto& [input, expected] = test_param;
    const auto& [A, b] = input;

    // Генерируем уникальное имя на основе размера и содержимого матрицы
    std::string test_name;

    if (A.empty()) {
      test_name = "empty_system";
    } else {
      size_t n = A.size();
      test_name = "system_" + std::to_string(n) + "x" + std::to_string(n);

      // Добавляем суффикс для уникальности на основе типа матрицы
      if (n == 1) {
        test_name += "_scalar";
      } else if (n == 2) {
        // Проверяем, диагональная ли матрица
        if (A[0][1] == 0.0 && A[1][0] == 0.0) {
          test_name += "_diagonal";
        } else {
          test_name += "_full";
        }
      } else if (n == 3) {
        // Добавляем идентификатор для разных тестов 3x3
        // Определяем тип матрицы по первому элементу
        if (A[0][0] == 2.0 && A[0][1] == 0.0) {
          test_name += "_diagonal";  // Тест 3
        } else if (A[0][0] == 4.0 && A[0][1] == 1.0) {
          test_name += "_spd";       // Тест 4
        } else if (A[0][0] == 2.0 && A[0][1] == -1.0) {
          test_name += "_tridiag";   // Тест 7
        }
      } else if (n == 4) {
        test_name += "_identity";
      }
    }

    return test_name;
  }

 private:
  // Проверяем, что A*x ≈ b
  bool CheckSolution(const std::vector<double>& x,
                     const std::vector<std::vector<double>>& A,
                     const std::vector<double>& b,
                     double tolerance = 1e-4) {
    if (A.empty() && x.empty() && b.empty()) {
      return true;
    }

    if (x.size() != b.size() || A.size() != b.size()) {
      return false;
    }

    // Вычисляем невязку r = b - A*x
    double max_residual = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
      double Ax_i = 0.0;
      for (size_t j = 0; j < A[i].size(); ++j) {
        Ax_i += A[i][j] * x[j];
      }
      double residual = std::abs(b[i] - Ax_i);
      if (residual > max_residual) {
        max_residual = residual;
      }
    }

    // Вычисляем норму b
    double b_norm = 0.0;
    for (double bi : b) {
      b_norm += bi * bi;
    }
    b_norm = std::sqrt(b_norm);

    if (b_norm < 1e-12) {
      // Если b близок к нулю, используем абсолютную погрешность
      return max_residual < tolerance;
    }

    // Используем относительную погрешность
    return max_residual / b_norm < tolerance;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto& [A, b] = input_data_;

    // Проверяем не решение, а то что A*x ≈ b
    return CheckSolution(output_data, A, b, 1e-4);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

// Вспомогательная функция для создания диагональной матрицы
std::vector<std::vector<double>> CreateDiagonalMatrix(int n, double value = 1.0) {
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i) {
    A[i][i] = value;
  }
  return A;
}

TEST_P(RemizovKRunFuncSystemLinearEquationsGradient, SolveLinearSystem) {
  ExecuteTest(GetParam());
}

// Упрощенные тесты с более реалистичными ожиданиями
const std::array<TestType, 8> kTestParam = {
    // Тест 1: Простейший случай 1x1
    std::make_tuple(
        std::make_tuple(
            std::vector<std::vector<double>>{{5.0}},  // A
            std::vector<double>{10.0}                 // b
        ),
        std::vector<double>{2.0}                      // ожидаемое x
    ),

    // Тест 2: Диагональная матрица 2x2 (простая)
    std::make_tuple(
        std::make_tuple(
            std::vector<std::vector<double>>{{2.0, 0.0}, {0.0, 3.0}},
            std::vector<double>{2.0, 3.0}  // b = diag(A) * [1, 1]
        ),
        std::vector<double>{1.0, 1.0}      // ожидаемое x = [1, 1]
    ),

    // Тест 3: Диагональная матрица 3x3
    std::make_tuple(
        std::make_tuple(
            CreateDiagonalMatrix(3, 2.0),
            std::vector<double>{2.0, 4.0, 6.0}  // b = 2 * [1, 2, 3]
        ),
        std::vector<double>{1.0, 2.0, 3.0}      // ожидаемое x = [1, 2, 3]
    ),

    // Тест 4: Симметричная положительно определенная матрица 3x3
    std::make_tuple(
        std::make_tuple(
            std::vector<std::vector<double>>{
                {4.0, 1.0, 1.0},
                {1.0, 3.0, 0.0},
                {1.0, 0.0, 2.0}
            },
            std::vector<double>{6.0, 4.0, 3.0}  // A * [1, 1, 1]
        ),
        std::vector<double>{1.0, 1.0, 1.0}      // ожидаемое x = [1, 1, 1]
    ),

    // Тест 5: Единичная матрица 4x4
    std::make_tuple(
        std::make_tuple(
            CreateDiagonalMatrix(4, 1.0),
            std::vector<double>{1.0, 2.0, 3.0, 4.0}
        ),
        std::vector<double>{1.0, 2.0, 3.0, 4.0}
    ),

    // Тест 6: Матрица с большими числами 2x2
    std::make_tuple(
        std::make_tuple(
            std::vector<std::vector<double>>{{100.0, 10.0}, {10.0, 100.0}},
            std::vector<double>{110.0, 110.0}  // A * [1, 1]
        ),
        std::vector<double>{1.0, 1.0}          // ожидаемое x = [1, 1]
    ),

    // Тест 7: Трехдиагональная матрица (симметричная) 3x3
    std::make_tuple(
        std::make_tuple(
            std::vector<std::vector<double>>{
                {2.0, -1.0, 0.0},
                {-1.0, 2.0, -1.0},
                {0.0, -1.0, 2.0}
            },
            std::vector<double>{1.0, 0.0, 1.0}
        ),
        std::vector<double>{0.75, 0.5, 0.75}   // точное решение
    ),

    // Тест 8: Пустая система (граничный случай)
    std::make_tuple(
        std::make_tuple(
            std::vector<std::vector<double>>{},  // A
            std::vector<double>{}                // b
        ),
        std::vector<double>{}                    // ожидаемое x
    )
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<RemizovKSystemLinearEquationsGradientMPI, InType>(
        kTestParam, PPC_SETTINGS_remizov_k_systems_linear_equations_gradient
    ),
    ppc::util::AddFuncTask<RemizovKSystemLinearEquationsGradientSEQ, InType>(
        kTestParam, PPC_SETTINGS_remizov_k_systems_linear_equations_gradient
    )
);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RemizovKRunFuncSystemLinearEquationsGradient::PrintFuncTestName<RemizovKRunFuncSystemLinearEquationsGradient>;

INSTANTIATE_TEST_SUITE_P(
    LinearSystemTests,
    RemizovKRunFuncSystemLinearEquationsGradient,
    kGtestValues,
    kPerfTestName
);

}  // namespace

}  // namespace remizov_k_max_in_matrix_string
