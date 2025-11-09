#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace remizov_k_max_in_matrix_string {

RemizovKMaxInMatrixStringMPI::RemizovKMaxInMatrixStringMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool RemizovKMaxInMatrixStringMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }
  if (!GetOutput().empty()) {
    return false;
  }

  const std::size_t row_size = GetInput()[0].size();
  return std::ranges::all_of(GetInput(), [row_size](const auto &row) { return row.size() == row_size; });
}

bool RemizovKMaxInMatrixStringMPI::PreProcessingImpl() {
  return true;
}

bool RemizovKMaxInMatrixStringMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int total_rows = 0;
  int row_size = 0;
  if (world_rank == 0) {
    total_rows = static_cast<int>(GetInput().size());
    if (total_rows > 0) {
      row_size = static_cast<int>(GetInput()[0].size());
    }
  }

  MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&row_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_rows == 0 || row_size == 0) {
    if (world_rank == 0) {
      GetOutput().resize(0);
    }
    return true;
  }

  std::vector<int> sendcounts(world_size, 0);
  std::vector<int> displs(world_size, 0);
  std::vector<int> continuous_data;

  if (world_rank == 0) {
    continuous_data.resize(static_cast<std::size_t>(total_rows) * static_cast<std::size_t>(row_size));
    for (int i = 0; i < total_rows; ++i) {
      for (int j = 0; j < row_size; ++j) {
        const std::size_t index =
            (static_cast<std::size_t>(i) * static_cast<std::size_t>(row_size)) + static_cast<std::size_t>(j);
        continuous_data[index] = GetInput()[i][j];
      }
    }

    int remaining = total_rows;
    for (int i = 0; i < world_size; ++i) {
      sendcounts[i] = (remaining + world_size - i - 1) / (world_size - i);
      remaining -= sendcounts[i];
      if (i > 0) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
      }
    }
    GetOutput().resize(static_cast<std::size_t>(total_rows));
  }

  MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  const int local_row_count = sendcounts[world_rank];

  std::vector<int> temp_sendcounts(world_size);
  std::vector<int> temp_displs(world_size);

  if (world_rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      temp_sendcounts[i] = sendcounts[i] * row_size;
      temp_displs[i] = displs[i] * row_size;
    }
  }

  MPI_Bcast(temp_sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(temp_displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_data;
  std::vector<int> local_maxes;

  if (local_row_count > 0) {
    local_data.resize(static_cast<std::size_t>(local_row_count) * static_cast<std::size_t>(row_size));

    const int *sendbuf = (world_rank == 0) ? continuous_data.data() : nullptr;
    MPI_Scatterv(sendbuf, temp_sendcounts.data(), temp_displs.data(), MPI_INT, local_data.data(),
                 local_row_count * row_size, MPI_INT, 0, MPI_COMM_WORLD);

    local_maxes.resize(static_cast<std::size_t>(local_row_count));
    for (int i = 0; i < local_row_count; ++i) {
      int max_val = std::numeric_limits<int>::min();
      for (int j = 0; j < row_size; ++j) {
        const std::size_t index =
            (static_cast<std::size_t>(i) * static_cast<std::size_t>(row_size)) + static_cast<std::size_t>(j);
        max_val = std::max(local_data[index], max_val);
      }
      local_maxes[static_cast<std::size_t>(i)] = max_val;
    }
  }

  int *recvbuf = (world_rank == 0) ? GetOutput().data() : nullptr;
  MPI_Gatherv(local_row_count > 0 ? local_maxes.data() : nullptr, local_row_count, MPI_INT, recvbuf, sendcounts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    return !GetOutput().empty();
  }
  return true;
}

bool RemizovKMaxInMatrixStringMPI::PostProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    for (const int value : GetOutput()) {
      if (value == std::numeric_limits<int>::min()) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace remizov_k_max_in_matrix_string
