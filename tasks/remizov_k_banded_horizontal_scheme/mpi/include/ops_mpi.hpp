#pragma once

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace remizov_k_banded_horizontal_scheme {

class RemizovKBandedHorizontalSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RemizovKBandedHorizontalSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace remizov_k_banded_horizontal_scheme
