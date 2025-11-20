#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <limits>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"

namespace remizov_k_max_in_matrix_string {

RemizovKMaxInMatrixStringMPI::RemizovKMaxInMatrixStringMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKMaxInMatrixStringMPI::ValidationImpl() {
  if (GetInput().empty()) return true;

  size_t first_row_size = GetInput()[0].size();
  for (const auto& row : GetInput()) {
    if (row.size() != first_row_size) return false;
  }
  return true;
}

bool RemizovKMaxInMatrixStringMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::vector<int> RemizovKMaxInMatrixStringMPI::CalculateRowDistribution(int world_size, int total_rows) {
  std::vector<int> rows_per_process(world_size);
  int base_rows = total_rows / world_size;
  int remainder = total_rows % world_size;

  for (int i = 0; i < world_size; ++i) {
    rows_per_process[i] = base_rows + (i < remainder ? 1 : 0);
  }
  return rows_per_process;
}

std::vector<int> RemizovKMaxInMatrixStringMPI::FindLocalMaxes(const std::vector<int>& local_data,
                                                             int local_rows, int row_size) {
  std::vector<int> local_maxes(local_rows);

  for (int i = 0; i < local_rows; ++i) {
    int max_val = std::numeric_limits<int>::min();
    for (int j = 0; j < row_size; ++j) {
      max_val = std::max(max_val, local_data[i * row_size + j]);
    }
    local_maxes[i] = max_val;
  }
  return local_maxes;
}

bool RemizovKMaxInMatrixStringMPI::RunImpl() {
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (GetInput().empty()) {
    if (world_rank == 0) GetOutput().clear();
    return true;
  }

  int total_rows = 0;
  int row_size = 0;

  if (world_rank == 0) {
    total_rows = static_cast<int>(GetInput().size());
    row_size = static_cast<int>(GetInput()[0].size());
  }

  MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&row_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_rows == 0 || row_size == 0) {
    if (world_rank == 0) GetOutput().clear();
    return true;
  }

  std::vector<int> rows_per_process = CalculateRowDistribution(world_size, total_rows);
  std::vector<int> displacements(world_size, 0);

  for (int i = 1; i < world_size; ++i) {
    displacements[i] = displacements[i-1] + rows_per_process[i-1];
  }

  int my_rows = rows_per_process[world_rank];

  std::vector<int> flat_matrix;
  if (world_rank == 0) {
    flat_matrix.resize(total_rows * row_size);
    for (int i = 0; i < total_rows; ++i) {
      for (int j = 0; j < row_size; ++j) {
        flat_matrix[i * row_size + j] = GetInput()[i][j];
      }
    }
  }

  std::vector<int> local_data(my_rows * row_size);
  std::vector<int> send_counts(world_size);
  std::vector<int> send_displs(world_size);

  for (int i = 0; i < world_size; ++i) {
    send_counts[i] = rows_per_process[i] * row_size;
    send_displs[i] = displacements[i] * row_size;
  }

  MPI_Scatterv(flat_matrix.data(), send_counts.data(), send_displs.data(),
              MPI_INT, local_data.data(), my_rows * row_size,
              MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_maxes = FindLocalMaxes(local_data, my_rows, row_size);

  std::vector<int> global_maxes;
  if (world_rank == 0) {
    global_maxes.resize(total_rows);
  }

  MPI_Gatherv(local_maxes.data(), my_rows, MPI_INT,
              global_maxes.data(), rows_per_process.data(), displacements.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = global_maxes;
  }

  return true;
}

bool RemizovKMaxInMatrixStringMPI::PostProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    for (int value : GetOutput()) {
      if (value == std::numeric_limits<int>::min()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace remizov_k_max_in_matrix_string
