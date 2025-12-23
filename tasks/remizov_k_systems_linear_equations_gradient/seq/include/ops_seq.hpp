#pragma once

#include "remizov_k_systems_linear_equations_gradient/common/include/common.hpp"
#include "task/include/task.hpp"

namespace remizov_k_systems_linear_equations_gradient {

class RemizovKSystemLinearEquationsGradientSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RemizovKSystemLinearEquationsGradientSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double DotProduct(const std::vector<double> &a, const std::vector<double> &b);
  std::vector<double> MatrixVectorMultiply(const std::vector<std::vector<double>> &A, const std::vector<double> &x);
  std::vector<double> VectorSubtract(const std::vector<double> &a, const std::vector<double> &b);
  std::vector<double> VectorAdd(const std::vector<double> &a, const std::vector<double> &b);
  std::vector<double> VectorScalarMultiply(const std::vector<double> &v, double scalar);
};

}  // namespace remizov_k_systems_linear_equations_gradient
