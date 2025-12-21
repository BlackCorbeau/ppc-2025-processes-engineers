#pragma once
#include <tuple>
#include <vector>

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

  void BroadcastMatrices(int rank);
  std::vector<int> CalculateRowRange(int size_procs, int rank, int total_rows);
  void MultiplyLocalRows(const Matrix &A_local, const Matrix &B, Matrix &C_local);
  void GatherResults(Matrix &C, const Matrix &C_local, int rank, int size);
};

}  // namespace remizov_k_banded_horizontal_scheme
