#pragma once
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"
#include "task/include/task.hpp"

namespace remizov_k_max_in_matrix_string {

class RemizovKSystemLinearEquationsGradientMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RemizovKSystemLinearEquationsGradientMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные функции
  double LocalDotProduct(const std::vector<double> &local_vec, const std::vector<double> &full_vec, int start_row,
                         int end_row);
  std::vector<double> LocalMatrixVectorMultiply(const std::vector<std::vector<double>> &local_A,
                                                const std::vector<double> &x, int n);
  void ConjugateGradientIteration(std::vector<double> &x, std::vector<double> &r, std::vector<double> &p, int start_row,
                                  int end_row, int rank, int size, int n);
  std::vector<std::pair<int, int>> CalculateRowRanges(int size, int n);
};

}  // namespace remizov_k_max_in_matrix_string
