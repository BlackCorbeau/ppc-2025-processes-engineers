#pragma once

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"
#include "task/include/task.hpp"

#include <vector>
#include <tuple>

namespace remizov_k_max_in_matrix_string {

class RemizovKMaxInMatrixStringMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RemizovKMaxInMatrixStringMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::tuple<int, int> BroadcastMatrixDimensions(int world_rank);
  bool ShouldEarlyReturn(int total_rows, int row_size, int world_rank);
  std::vector<int> FlattenInputData(int total_rows, int row_size);
  void CalculateDataDistribution(std::vector<int>& sendcounts, std::vector<int>& displs,
                                 int total_rows, int world_size);
  std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
  PrepareDataDistribution(int world_rank, int world_size, int total_rows, int row_size);
  void BroadcastDistributionInfo(std::vector<int>& sendcounts, std::vector<int>& displs,
                                 int world_size);
  std::tuple<std::vector<int>, std::vector<int>>
  PrepareScatterParameters(const std::vector<int>& sendcounts, const std::vector<int>& displs,
                           int row_size, int world_rank, int world_size);
  std::vector<int> CalculateLocalMaxes(const std::vector<int>& local_data,
                                       int local_row_count, int row_size);
  std::vector<int> ProcessLocalData(int world_rank, int local_row_count, int row_size,
                                    const std::vector<int>& continuous_data, const std::vector<int>& temp_sendcounts,
                                    const std::vector<int>& temp_displs);
  void GatherResults(const std::vector<int>& local_maxes, const std::vector<int>& sendcounts,
                     const std::vector<int>& displs, int world_rank);
  bool Finalize(int world_rank);
};

}  // namespace remizov_k_max_in_matrix_string
