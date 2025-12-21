#pragma once

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace remizov_k_banded_horizontal_scheme {

class RemizovKBandedHorizontalSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RemizovKBandedHorizontalSchemeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static bool AreMatricesCompatible(const Matrix &A, const Matrix &B);
  static Matrix MultiplyMatrices(const Matrix &A, const Matrix &B);
};

}  // namespace remizov_k_banded_horizontal_scheme
