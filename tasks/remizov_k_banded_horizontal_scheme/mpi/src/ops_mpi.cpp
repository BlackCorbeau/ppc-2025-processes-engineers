#include "remizov_k_banded_horizontal_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "remizov_k_banded_horizontal_scheme/common/include/common.hpp"

namespace remizov_k_banded_horizontal_scheme {

namespace {

bool AreRowsConsistent(const Matrix &matrix) {
  if (matrix.empty()) {
    return true;
  }

  const size_t first_row_size = matrix[0].size();
  return std::all_of(matrix.begin(), matrix.end(),
                     [first_row_size](const auto &row) { return row.size() == first_row_size; });
}

void BroadcastSizesAndResizeMatrices(int rank, InType &input_data, std::array<int, 4> &sizes) {
  auto &[a, b] = input_data;

  if (rank == 0) {
    sizes[0] = static_cast<int>(a.size());
    sizes[1] = a.empty() ? 0 : static_cast<int>(a[0].size());
    sizes[2] = static_cast<int>(b.size());
    sizes[3] = b.empty() ? 0 : static_cast<int>(b[0].size());
  }

  MPI_Bcast(sizes.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    if (sizes[0] > 0 && sizes[1] > 0) {
      a.resize(sizes[0], std::vector<int>(sizes[1]));
    } else {
      a.clear();
    }
    if (sizes[2] > 0 && sizes[3] > 0) {
      b.resize(sizes[2], std::vector<int>(sizes[3]));
    } else {
      b.clear();
    }
  }
}

void BroadcastMatrixData(const std::array<int, 4> &sizes, InType &input_data) {
  auto &[a, b] = input_data;

  // Broadcast matrix b
  if (sizes[2] > 0 && sizes[3] > 0) {
    for (int i = 0; i < sizes[2]; ++i) {
      MPI_Bcast(b[i].data(), sizes[3], MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  // Broadcast matrix a
  if (sizes[0] > 0 && sizes[1] > 0) {
    for (int i = 0; i < sizes[0]; ++i) {
      MPI_Bcast(a[i].data(), sizes[1], MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

void BroadcastMatricesImpl(int rank, InType &input_data) {
  std::array<int, 4> sizes{};
  BroadcastSizesAndResizeMatrices(rank, input_data, sizes);
  BroadcastMatrixData(sizes, input_data);
}

std::vector<int> CalculateRowRangeImpl(int size_procs, int rank, int total_rows) {
  if (total_rows <= 0 || size_procs <= 0 || rank < 0 || rank >= size_procs) {
    return {-1, -1};
  }

  const int rows_per_proc = total_rows / size_procs;
  const int remainder = total_rows % size_procs;
  int start = 0;
  int end = 0;

  if (rank < remainder) {
    start = rank * (rows_per_proc + 1);
    end = start + rows_per_proc;
  } else {
    start = (remainder * (rows_per_proc + 1)) + ((rank - remainder) * rows_per_proc);
    end = start + rows_per_proc - 1;
  }

  if (start >= total_rows) {
    return {-1, -1};
  }

  if (end >= total_rows) {
    end = total_rows - 1;
  }

  return {start, end};
}

Matrix MultiplyLocalRowsImpl(const Matrix &a_local, const Matrix &b) {
  if (a_local.empty() || b.empty() || a_local[0].size() != b.size()) {
    return {};
  }

  const size_t n_local = a_local.size();
  const size_t m = a_local[0].size();
  const size_t p = b[0].size();

  Matrix c_local(n_local, std::vector<int>(p, 0));

  for (size_t i = 0; i < n_local; ++i) {
    for (size_t j = 0; j < p; ++j) {
      int sum = 0;
      for (size_t k = 0; k < m; ++k) {
        sum += a_local[i][k] * b[k][j];
      }
      c_local[i][j] = sum;
    }
  }

  return c_local;
}

void GatherColumnData(int rank, int rows_local, const Matrix &c_local, const std::vector<int> &all_rows,
                      Matrix *result_ptr) {
  if (rank != 0) {
    if (rows_local > 0 && !c_local.empty()) {
      const int cols = static_cast<int>(c_local[0].size());
      for (int col = 0; col < cols; ++col) {
        std::vector<int> local_col(rows_local);
        for (int i = 0; i < rows_local; ++i) {
          local_col[i] = c_local[i][col];
        }
        MPI_Gatherv(local_col.data(), rows_local, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
      }
    } else {
      MPI_Gatherv(nullptr, 0, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return;
  }

  const int cols = rows_local > 0 ? static_cast<int>(c_local[0].size()) : 0;
  if (cols <= 0 || result_ptr == nullptr) {
    return;
  }

  Matrix &result = *result_ptr;
  std::vector<int> displs(all_rows.size(), 0);
  for (size_t i = 1; i < all_rows.size(); ++i) {
    displs[i] = displs[i - 1] + all_rows[i - 1];
  }

  for (int col = 0; col < cols; ++col) {
    std::vector<int> local_col(rows_local);
    for (int i = 0; i < rows_local; ++i) {
      local_col[i] = c_local[i][col];
    }

    std::vector<int> global_col(result.size());
    MPI_Gatherv(local_col.data(), rows_local, MPI_INT, global_col.data(), all_rows.data(), displs.data(), MPI_INT, 0,
                MPI_COMM_WORLD);

    for (size_t i = 0; i < result.size(); ++i) {
      result[i][col] = global_col[i];
    }
  }
}

Matrix GatherResultsImpl(const Matrix &c_local, int rank, int size) {
  const int rows_local = static_cast<int>(c_local.size());
  std::vector<int> all_rows(size, 0);

  MPI_Gather(&rows_local, 1, MPI_INT, all_rows.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int total_rows = 0;
    for (int rows : all_rows) {
      total_rows += rows;
    }

    if (total_rows <= 0) {
      return {};
    }

    const int cols = rows_local > 0 ? static_cast<int>(c_local[0].size()) : 0;
    if (cols <= 0) {
      return {};
    }

    Matrix result(total_rows, std::vector<int>(cols, 0));
    GatherColumnData(rank, rows_local, c_local, all_rows, &result);
    return result;
  }

  // Non-zero ranks pass nullptr for result
  GatherColumnData(rank, rows_local, c_local, all_rows, nullptr);
  return {};
}

void BroadcastResult(Matrix &result, int rank) {
  int result_rows = 0;
  int result_cols = 0;

  if (rank == 0) {
    result_rows = static_cast<int>(result.size());
    result_cols = result.empty() ? 0 : static_cast<int>(result[0].size());
  }

  MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (result_rows <= 0 || result_cols <= 0) {
    if (rank != 0) {
      result.clear();
    }
    return;
  }

  if (rank != 0) {
    result.resize(result_rows, std::vector<int>(result_cols));
  }

  for (auto &row : result) {
    MPI_Bcast(row.data(), result_cols, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

}  // namespace

RemizovKBandedHorizontalSchemeMPI::RemizovKBandedHorizontalSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKBandedHorizontalSchemeMPI::ValidationImpl() {
  const auto &[a, b] = GetInput();

  if (a.empty() || b.empty()) {
    return false;
  }

  return AreRowsConsistent(a) && AreRowsConsistent(b);
}

bool RemizovKBandedHorizontalSchemeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RemizovKBandedHorizontalSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &input_data = GetInput();
  BroadcastMatricesImpl(rank, input_data);

  const auto &[a, b] = input_data;

  if (a.empty() || b.empty() || a[0].size() != b.size()) {
    if (rank == 0) {
      throw std::invalid_argument("Matrices are not compatible for multiplication");
    }
    return false;
  }

  const std::vector<int> row_range = CalculateRowRangeImpl(size, rank, static_cast<int>(a.size()));

  Matrix a_local;
  if (row_range[0] >= 0 && row_range[0] <= row_range[1] && row_range[1] < static_cast<int>(a.size())) {
    a_local.reserve(row_range[1] - row_range[0] + 1);
    for (int i = row_range[0]; i <= row_range[1]; ++i) {
      a_local.push_back(a[i]);
    }
  }

  const Matrix c_local = MultiplyLocalRowsImpl(a_local, b);
  Matrix result = GatherResultsImpl(c_local, rank, size);

  BroadcastResult(result, rank);

  if (rank == 0 || !result.empty()) {
    GetOutput() = result;
  }

  return true;
}

bool RemizovKBandedHorizontalSchemeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_banded_horizontal_scheme
