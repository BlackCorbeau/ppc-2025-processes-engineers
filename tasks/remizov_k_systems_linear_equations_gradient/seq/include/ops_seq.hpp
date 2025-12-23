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

  static bool IsSystemCompatible(const std::vector<std::vector<double>> &A, const std::vector<double> &b);
  static double DotProduct(const std::vector<double> &a, const std::vector<double> &b);
  static std::vector<double> MatrixVectorMultiply(const std::vector<std::vector<double>> &A, const std::vector<double> &x);
  static std::vector<double> VectorSubtract(const std::vector<double> &a, const std::vector<double> &b);
  static std::vector<double> VectorAdd(const std::vector<double> &a, const std::vector<double> &b);
  static std::vector<double> VectorScalarMultiply(const std::vector<double> &v, double scalar);
  static std::vector<double> SolveSystem(const std::vector<std::vector<double>> &A, const std::vector<double> &b,
                                        double tolerance = 1e-10, int max_iterations = 1000);
};

}  // namespace remizov_k_systems_linear_equations_gradient
