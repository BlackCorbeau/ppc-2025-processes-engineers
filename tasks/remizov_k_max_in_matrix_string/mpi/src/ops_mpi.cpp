#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"

namespace remizov_k_max_in_matrix_string {

namespace {

bool IsSystemCompatible(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector_b) {
  if (matrix.empty() && vector_b.empty()) {
    return true;
  }

  if (matrix.empty() || vector_b.empty()) {
    return false;
  }

  const size_t n = matrix.size();
  if (n != vector_b.size()) {
    return false;
  }

  return std::ranges::all_of(matrix, [n](const auto &row) { return row.size() == n; });
}

void BroadcastSystemData(int rank, InType &input_data) {
  auto &[matrix, vector_b] = input_data;

  int is_empty = 0;
  if (rank == 0) {
    is_empty = (matrix.empty() && vector_b.empty()) ? 1 : 0;
  }
  MPI_Bcast(&is_empty, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (is_empty != 0) {
    return;
  }

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(matrix.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0) {
    return;
  }

  if (rank != 0) {
    matrix.resize(static_cast<size_t>(n));
    for (auto &row : matrix) {
      row.resize(static_cast<size_t>(n), 0.0);
    }
    vector_b.resize(static_cast<size_t>(n), 0.0);
  }

  for (int i = 0; i < n; ++i) {
    MPI_Bcast(matrix[static_cast<size_t>(i)].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(vector_b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

std::vector<int> CalculateRowRange(int size_procs, int rank, int total_rows) {
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

void PerformCGIteration(int n, const std::vector<std::vector<double>> &matrix, std::vector<double> &x,
                        std::vector<double> &r, std::vector<double> &p, std::vector<double> &ap, double &r_sq,
                        int &iteration) {
  std::ranges::fill(ap, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      ap[i] += matrix[i][j] * p[j];
    }
  }

  double p_ap = 0.0;
  for (int i = 0; i < n; ++i) {
    p_ap += p[i] * ap[i];
  }

  if (std::abs(p_ap) < 1e-15) {
    p_ap = r_sq;
  }

  const double alpha = r_sq / p_ap;

  for (int i = 0; i < n; ++i) {
    x[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
  }

  double r_sq_new = 0.0;
  for (int i = 0; i < n; ++i) {
    r_sq_new += r[i] * r[i];
  }

  const double beta = r_sq_new / r_sq;

  for (int i = 0; i < n; ++i) {
    p[i] = r[i] + (beta * p[i]);
  }

  r_sq = r_sq_new;
  iteration++;
}

std::vector<double> SolveSimpleSystem(int rank, int n, const std::vector<std::vector<double>> &matrix,
                                      const std::vector<double> &vector_b) {
  std::vector<double> x(n, 0.0);

  if (rank == 0) {
    const double tolerance = 1e-6;
    const int max_iterations = n * 2;

    std::vector<double> r(n);
    std::vector<double> p(n);
    std::vector<double> ap(n);

    std::ranges::copy(vector_b, r.begin());
    std::ranges::copy(r, p.begin());

    double r_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      r_sq += r[i] * r[i];
    }

    int iteration = 0;

    while (iteration < max_iterations && std::sqrt(r_sq) > tolerance) {
      PerformCGIteration(n, matrix, x, r, p, ap, r_sq, iteration);
    }
  }

  MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return x;
}

void CollectLocalData(int rank, int size, int n, const std::vector<int> &range, const std::vector<double> &local_data,
                      std::vector<double> &global_data) {
  const int start_row = range[0];
  const int local_rows = range[1] - start_row + 1;

  if (rank == 0) {
    for (int i = 0; i < local_rows; ++i) {
      global_data[start_row + i] = local_data[i];
    }

    for (int proc = 1; proc < size; ++proc) {
      const auto proc_range = CalculateRowRange(size, proc, n);
      const int proc_start = proc_range[0];
      const int proc_rows = proc_range[1] - proc_start + 1;

      std::vector<double> recv_buf(proc_rows);
      MPI_Recv(recv_buf.data(), proc_rows, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < proc_rows; ++i) {
        global_data[proc_start + i] = recv_buf[i];
      }
    }
  } else {
    MPI_Send(local_data.data(), local_rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

void UpdateGlobalP(int rank, int size, int n, const std::vector<int> &range, const std::vector<double> &local_p,
                   std::vector<double> &global_p) {
  CollectLocalData(rank, size, n, range, local_p, global_p);
  MPI_Bcast(global_p.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void ComputeLocalMatrixVectorProduct(int start_row, int local_rows, int n,
                                     const std::vector<std::vector<double>> &matrix,
                                     const std::vector<double> &global_p, std::vector<double> &local_ap) {
  std::ranges::fill(local_ap, 0.0);
  for (int i = 0; i < local_rows; ++i) {
    const int global_row = start_row + i;
    for (int j = 0; j < n; ++j) {
      local_ap[i] += matrix[global_row][j] * global_p[j];
    }
  }
}

void PerformMPICGIteration(int rank, int size, int start_row, int local_rows, int n,
                           const std::vector<std::vector<double>> &matrix, const std::vector<int> &range,
                           std::vector<double> &local_x, std::vector<double> &local_r, std::vector<double> &local_p,
                           std::vector<double> &local_ap, std::vector<double> &global_p, double &global_r_sq,
                           int &iteration) {
  ComputeLocalMatrixVectorProduct(start_row, local_rows, n, matrix, global_p, local_ap);

  double p_ap_local = 0.0;
  for (int i = 0; i < local_rows; ++i) {
    p_ap_local += local_p[i] * local_ap[i];
  }

  double p_ap_global = 0.0;
  MPI_Allreduce(&p_ap_local, &p_ap_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (std::abs(p_ap_global) < 1e-15) {
    return;
  }

  const double alpha = global_r_sq / p_ap_global;

  for (int i = 0; i < local_rows; ++i) {
    local_x[i] += alpha * local_p[i];
    local_r[i] -= alpha * local_ap[i];
  }

  double r_sq_new_local = 0.0;
  for (int i = 0; i < local_rows; ++i) {
    r_sq_new_local += local_r[i] * local_r[i];
  }

  double global_r_sq_new = 0.0;
  MPI_Allreduce(&r_sq_new_local, &global_r_sq_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const double beta = global_r_sq_new / global_r_sq;

  for (int i = 0; i < local_rows; ++i) {
    local_p[i] = local_r[i] + (beta * local_p[i]);
  }

  UpdateGlobalP(rank, size, n, range, local_p, global_p);

  global_r_sq = global_r_sq_new;
  iteration++;
}

std::vector<double> SolveSystemMPI(int rank, int size, const std::vector<std::vector<double>> &matrix,
                                   const std::vector<double> &vector_b) {
  if (matrix.empty() && vector_b.empty()) {
    return {};
  }

  const int n = static_cast<int>(vector_b.size());

  if (n <= 2 || size > n) {
    return SolveSimpleSystem(rank, n, matrix, vector_b);
  }

  const auto range = CalculateRowRange(size, rank, n);
  const int start_row = range[0];
  const int end_row = range[1];

  if (start_row < 0 || end_row < 0) {
    return {};
  }

  const int local_rows = end_row - start_row + 1;

  const double tolerance = 1e-6;
  const int max_iterations = n * 2;

  std::vector<double> local_x(local_rows, 0.0);
  std::vector<double> local_r(local_rows);
  std::vector<double> local_p(local_rows);
  std::vector<double> local_ap(local_rows);

  for (int i = 0; i < local_rows; ++i) {
    const int global_idx = start_row + i;
    local_r[i] = vector_b[global_idx];
    local_p[i] = local_r[i];
  }

  std::vector<double> global_x(n, 0.0);
  std::vector<double> global_p(n);

  UpdateGlobalP(rank, size, n, range, local_p, global_p);

  double r_sq = 0.0;
  for (int i = 0; i < local_rows; ++i) {
    r_sq += local_r[i] * local_r[i];
  }

  double global_r_sq = 0.0;
  MPI_Allreduce(&r_sq, &global_r_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int iteration = 0;

  while (iteration < max_iterations && std::sqrt(global_r_sq) > tolerance) {
    PerformMPICGIteration(rank, size, start_row, local_rows, n, matrix, range, local_x, local_r, local_p, local_ap,
                          global_p, global_r_sq, iteration);
  }

  CollectLocalData(rank, size, n, range, local_x, global_x);
  MPI_Bcast(global_x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return global_x;
}

}  // namespace

RemizovKSystemLinearEquationsGradientMPI::RemizovKSystemLinearEquationsGradientMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKSystemLinearEquationsGradientMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid = true;

  if (rank == 0) {
    const auto &[matrix, vector_b] = GetInput();

    if (matrix.empty() && vector_b.empty()) {
      is_valid = true;
    } else if (matrix.empty() || vector_b.empty()) {
      is_valid = false;
    } else {
      is_valid = IsSystemCompatible(matrix, vector_b);
    }
  }

  int valid_int = is_valid ? 1 : 0;
  MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return valid_int == 1;
}

bool RemizovKSystemLinearEquationsGradientMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RemizovKSystemLinearEquationsGradientMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &input_data = GetInput();
  const auto &[matrix, vector_b] = input_data;

  if (matrix.empty() && vector_b.empty()) {
    GetOutput() = std::vector<double>();
    return true;
  }

  BroadcastSystemData(rank, input_data);

  std::vector<double> solution = SolveSystemMPI(rank, size, matrix, vector_b);

  GetOutput() = solution;

  return true;
}

bool RemizovKSystemLinearEquationsGradientMPI::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_max_in_matrix_string
