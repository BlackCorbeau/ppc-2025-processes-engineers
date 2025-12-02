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

  if (start < 0 || end < 0 || start > end || start >= static_cast<int>(GetInput().size())) {
    return result;  // Возвращаем пустой вектор
  }

  int actual_end = end;
  if (end >= static_cast<int>(GetInput().size())) {
    actual_end = static_cast<int>(GetInput().size()) - 1;
  }

  for (int i = start; i <= actual_end; i++) {
    if (!GetInput()[i].empty()) {
      int max_val = *std::max_element(GetInput()[i].begin(), GetInput()[i].end());
      result.push_back(max_val);
    } else {
      result.push_back(std::numeric_limits<int>::min());
    }
  }
  return result;
}

std::vector<int> RemizovKMaxInMatrixStringMPI::CalculatingInterval(int size_prcs, int rank, int count_rows) {
  std::vector<int> interval(2, -1);

  if (count_rows <= 0 || size_prcs <= 0 || rank < 0 || rank >= size_prcs) {
    return interval;
  }

  int rows_per_process = count_rows / size_prcs;
  int remainder = count_rows % size_prcs;

  if (rank < remainder) {
    interval[0] = rank * (rows_per_process + 1);
    interval[1] = interval[0] + rows_per_process;  // не -1, потому что мы уже включили одну
  } else {
    interval[0] = remainder * (rows_per_process + 1) + (rank - remainder) * rows_per_process;
    interval[1] = interval[0] + rows_per_process - 1;
  }

  if (interval[0] >= count_rows) {
    interval[0] = -1;
    interval[1] = -1;
  }
  if (interval[1] >= count_rows) {
    interval[1] = count_rows - 1;
  }

  return interval;
}

bool RemizovKMaxInMatrixStringMPI::RunImpl() {
  if (GetInput().empty()) {
    return true;
  }

  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int count_rows = static_cast<int>(GetInput().size());

  if (rank == 0) {
    for (int i = 1; i < size; i++) {
      std::vector<int> interval = CalculatingInterval(size, i, count_rows);
      MPI_Send(interval.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    std::vector<int> interval = CalculatingInterval(size, 0, count_rows);
    std::vector<int> max_values;

    if (interval[0] != -1 && interval[1] != -1) {
      max_values = FindMaxValues(interval[0], interval[1]);
    }

    for (int &value : max_values) {
      GetOutput().push_back(value);
    }

    MPI_Status status;
    for (int i = 1; i < size; i++) {
      int size_values = 0;
      MPI_Recv(&size_values, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);

      if (size_values > 0) {
        std::vector<int> buf(size_values);
        MPI_Recv(buf.data(), size_values, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
        for (int &value : buf) {
          GetOutput().push_back(value);
        }
      }
    }

  } else {
    MPI_Status status;
    std::vector<int> buf(2);
    MPI_Recv(buf.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    std::vector<int> max_values;
    if (buf[0] != -1 && buf[1] != -1) {
      max_values = FindMaxValues(buf[0], buf[1]);
    }

    int size_values = static_cast<int>(max_values.size());
    MPI_Send(&size_values, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

    if (size_values > 0) {
      MPI_Send(max_values.data(), size_values, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
  }

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(total_size);
  }

  if (total_size > 0) {
    MPI_Bcast(GetOutput().data(), total_size, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool RemizovKMaxInMatrixStringMPI::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_max_in_matrix_string
