#include "remizov_k_banded_horizontal_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"

namespace remizov_k_banded_horizontal_scheme {

RemizovKBandedHorizontalSchemeMPI::RemizovKBandedHorizontalSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKBandedHorizontalSchemeMPI::ValidationImpl() {
  const auto &[A, B] = GetInput();

  if (A.empty() || B.empty()) {
    return false;
  }

  size_t cols_A = A[0].size();
  for (const auto &row : A) {
    if (row.size() != cols_A) {
      return false;
    }
  }

  size_t cols_B = B[0].size();
  for (const auto &row : B) {
    if (row.size() != cols_B) {
      return false;
    }
  }

  return true;
}

bool RemizovKBandedHorizontalSchemeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void RemizovKBandedHorizontalSchemeMPI::BroadcastMatrices(int rank) {
  const auto &[A, B] = GetInput();
  int sizes[4];  // nA, mA, nB, mB

  if (rank == 0) {
    sizes[0] = static_cast<int>(A.size());
    sizes[1] = A.empty() ? 0 : static_cast<int>(A[0].size());
    sizes[2] = static_cast<int>(B.size());
    sizes[3] = B.empty() ? 0 : static_cast<int>(B[0].size());
  }

  MPI_Bcast(sizes, 4, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    auto &[A_mod, B_mod] = GetInput();
    if (sizes[0] > 0 && sizes[1] > 0) {
      A_mod.resize(sizes[0], std::vector<int>(sizes[1]));
    } else {
      A_mod.clear();
    }
    if (sizes[2] > 0 && sizes[3] > 0) {
      B_mod.resize(sizes[2], std::vector<int>(sizes[3]));
    } else {
      B_mod.clear();
    }
  }

  // Broadcast matrix B
  if (sizes[2] > 0 && sizes[3] > 0) {
    auto &[A_mod, B_mod] = GetInput();
    for (int i = 0; i < sizes[2]; ++i) {
      MPI_Bcast(B_mod[i].data(), sizes[3], MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  // Broadcast matrix A
  if (sizes[0] > 0 && sizes[1] > 0) {
    auto &[A_mod, B_mod] = GetInput();
    for (int i = 0; i < sizes[0]; ++i) {
      MPI_Bcast(A_mod[i].data(), sizes[1], MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

std::vector<int> RemizovKBandedHorizontalSchemeMPI::CalculateRowRange(int size_procs, int rank, int total_rows) {
  if (total_rows <= 0 || size_procs <= 0 || rank < 0 || rank >= size_procs) {
    return {-1, -1};
  }

  int rows_per_proc = total_rows / size_procs;
  int remainder = total_rows % size_procs;

  int start = 0;
  int end = 0;

  if (rank < remainder) {
    start = rank * (rows_per_proc + 1);
    end = start + rows_per_proc;
  } else {
    start = (remainder * (rows_per_proc + 1)) + ((rank - remainder) * rows_per_proc);
    end = start + rows_per_proc - 1;
  }

  // Ensure bounds are valid
  if (start >= total_rows) {
    return {-1, -1};
  }

  if (end >= total_rows) {
    end = total_rows - 1;
  }

  return {start, end};
}

void RemizovKBandedHorizontalSchemeMPI::MultiplyLocalRows(const Matrix &A_local, const Matrix &B, Matrix &C_local) {
  if (A_local.empty() || B.empty() || A_local[0].size() != B.size()) {
    C_local.clear();
    return;
  }

  size_t n_local = A_local.size();
  size_t m = A_local[0].size();
  size_t p = B[0].size();

  C_local.resize(n_local, std::vector<int>(p, 0));

  for (size_t i = 0; i < n_local; ++i) {
    for (size_t j = 0; j < p; ++j) {
      int sum = 0;
      for (size_t k = 0; k < m; ++k) {
        sum += A_local[i][k] * B[k][j];
      }
      C_local[i][j] = sum;
    }
  }
}

void RemizovKBandedHorizontalSchemeMPI::GatherResults(Matrix &C, const Matrix &C_local, int rank, int size) {
  int rows_local = static_cast<int>(C_local.size());
  std::vector<int> all_rows(size, 0);

  MPI_Gather(&rows_local, 1, MPI_INT, all_rows.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int total_rows = 0;
    for (int rows : all_rows) {
      total_rows += rows;
    }

    if (total_rows > 0) {
      int cols = rows_local > 0 ? static_cast<int>(C_local[0].size()) : 0;
      if (cols > 0) {
        C.resize(total_rows);
        for (auto &row : C) {
          row.resize(cols, 0);
        }

        std::vector<int> displs(size, 0);
        for (int i = 1; i < size; ++i) {
          displs[i] = displs[i - 1] + all_rows[i - 1];
        }

        for (int col = 0; col < cols; ++col) {
          std::vector<int> local_col(rows_local);
          for (int i = 0; i < rows_local; ++i) {
            local_col[i] = C_local[i][col];
          }

          std::vector<int> global_col(total_rows);
          MPI_Gatherv(local_col.data(), rows_local, MPI_INT, global_col.data(), all_rows.data(), displs.data(), MPI_INT,
                      0, MPI_COMM_WORLD);

          for (int i = 0; i < total_rows; ++i) {
            C[i][col] = global_col[i];
          }
        }
      } else {
        C.clear();
      }
    } else {
      C.clear();
    }
  } else {
    if (rows_local > 0 && !C_local.empty()) {
      int cols = static_cast<int>(C_local[0].size());

      for (int col = 0; col < cols; ++col) {
        std::vector<int> local_col(rows_local);
        for (int i = 0; i < rows_local; ++i) {
          local_col[i] = C_local[i][col];
        }

        MPI_Gatherv(local_col.data(), rows_local, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
      }
    } else {
      // Просто собираем информацию о нулевых строках
      MPI_Gatherv(nullptr, 0, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

bool RemizovKBandedHorizontalSchemeMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  BroadcastMatrices(rank);

  const auto &[A, B] = GetInput();

  if (A.empty() || B.empty() || A[0].size() != B.size()) {
    if (rank == 0) {
      throw std::invalid_argument("Matrices are not compatible for multiplication");
    }
    return false;
  }

  std::vector<int> row_range = CalculateRowRange(size, rank, static_cast<int>(A.size()));

  Matrix A_local;
  if (row_range[0] >= 0 && row_range[0] <= row_range[1] && row_range[1] < static_cast<int>(A.size())) {
    A_local.reserve(row_range[1] - row_range[0] + 1);
    for (int i = row_range[0]; i <= row_range[1]; ++i) {
      A_local.push_back(A[i]);
    }
  }

  Matrix C_local;
  MultiplyLocalRows(A_local, B, C_local);

  Matrix C;
  GatherResults(C, C_local, rank, size);

  if (rank == 0) {
    GetOutput() = C;

    int result_rows = static_cast<int>(C.size());
    int result_cols = !C.empty() ? static_cast<int>(C[0].size()) : 0;

    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_rows > 0 && result_cols > 0) {
      for (auto &row : C) {
        MPI_Bcast(row.data(), result_cols, MPI_INT, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    int result_rows, result_cols;
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_rows > 0 && result_cols > 0) {
      Matrix C_recv(result_rows, std::vector<int>(result_cols, 0));
      for (auto &row : C_recv) {
        MPI_Bcast(row.data(), result_cols, MPI_INT, 0, MPI_COMM_WORLD);
      }
      GetOutput() = C_recv;
    } else {
      GetOutput().clear();
    }
  }

  return true;
}

bool RemizovKBandedHorizontalSchemeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_banded_horizontal_scheme
