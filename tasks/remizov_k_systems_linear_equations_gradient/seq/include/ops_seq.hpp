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
};

}  // namespace remizov_k_systems_linear_equations_gradient
