#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"

namespace remizov_k_max_in_matrix_string {

RemizovKMaxInMatrixStringMPI::RemizovKMaxInMatrixStringMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKMaxInMatrixStringMPI::ValidationImpl() {
  return true;
}

bool RemizovKMaxInMatrixStringMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::vector<int> RemizovKMaxInMatrixStringMPI::FindMaxValues(const int start, const int end) {
  std::vector<int> result;
  for (int i = start; i <= end; i++) {
    if (!GetInput()[i].empty()) {
      int max_val = *std::max_element(GetInput()[i].begin(), GetInput()[i].end());
      result.push_back(max_val);
    }
  }
  return result;
}

std::vector<int> RemizovKMaxInMatrixStringMPI::CalculatingInterval(int size_prcs, int rank, int count_rows) {
  std::vector<int> vec(2);
  int whole_part = count_rows / size_prcs;
  int real_part = count_rows % size_prcs;
  int start = rank * whole_part;
  if ((rank - 1 < real_part) && (rank - 1 != -1)) {
    start += rank;
  } else if (rank != 0) {
    start += real_part;
  }
  int end = start + whole_part - 1;
  if (rank < real_part) {
    end += 1;
  }
  vec[0] = start;
  vec[1] = end;
  return vec;
}

bool RemizovKMaxInMatrixStringMPI::RunImpl() {
  if (GetInput().empty()) {
    return true;
  }

  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int count_rows = static_cast<int>(GetInput().size());

    for (int i = 1; i < size; i++) {
      std::vector<int> interval = CalculatingInterval(size, i, count_rows);
      MPI_Send(interval.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    std::vector<int> interval = CalculatingInterval(size, 0, count_rows);
    std::vector<int> max_values = FindMaxValues(interval[0], interval[1]);
    for (int &value : max_values) {
      GetOutput().push_back(value);
    }

    MPI_Status status;
    for (int i = 1; i < size; i++) {
      int size_values = 0;
      MPI_Recv(&size_values, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
      std::vector<int> buf(size_values);
      MPI_Recv(buf.data(), size_values, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
      for (int &value : buf) {
        GetOutput().push_back(value);
      }
    }

  } else {
    MPI_Status status;
    std::vector<int> buf(2);
    MPI_Recv(buf.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    std::vector<int> max_values = FindMaxValues(buf[0], buf[1]);
    int size_values = static_cast<int>(max_values.size());
    MPI_Send(&size_values, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(max_values.data(), size_values, MPI_INT, 0, 2, MPI_COMM_WORLD);
  }

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(total_size);
  }

  MPI_Bcast(GetOutput().data(), total_size, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool RemizovKMaxInMatrixStringMPI::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_max_in_matrix_string
