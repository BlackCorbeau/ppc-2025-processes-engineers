#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"

namespace remizov_k_max_in_matrix_string {

namespace {

const double kDefaultTolerance = 1e-10;
const int kDefaultMaxIterations = 1000;

bool IsSystemCompatible(const std::vector<std::vector<double>> &A, const std::vector<double> &b) {
  if (A.empty() && b.empty()) {
    return true;
  }

  if (A.empty() || b.empty()) {
    return false;
  }

  size_t n = A.size();
  if (n != b.size()) {
    return false;
  }

  for (const auto &row : A) {
    if (row.size() != n) {
      return false;
    }
  }

  return true;
}

void BroadcastSystemData(int rank, InType &input_data) {
  auto &[A, b] = input_data;

  int is_empty = 0;
  if (rank == 0) {
    is_empty = (A.empty() && b.empty()) ? 1 : 0;
  }
  MPI_Bcast(&is_empty, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (is_empty) {
    return;
  }

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(A.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0) {
    return;
  }

  if (rank != 0) {
    A.resize(static_cast<size_t>(n));
    for (auto &row : A) {
      row.resize(static_cast<size_t>(n), 0.0);
    }
    b.resize(static_cast<size_t>(n), 0.0);
  }

  for (int i = 0; i < n; ++i) {
    MPI_Bcast(A[static_cast<size_t>(i)].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

std::vector<double> SolveSystemMPI(int rank, int size, const std::vector<std::vector<double>> &A,
                                   const std::vector<double> &b) {
  if (A.empty() && b.empty()) {
    return std::vector<double>();
  }

  int n = static_cast<int>(b.size());

  if (n <= 2 || size > n) {
    std::vector<double> x(n, 0.0);

    if (rank == 0) {
      const double TOLERANCE = 1e-6;
      const int MAX_ITERATIONS = n * 2;

      std::vector<double> r(n);
      std::vector<double> p(n);
      std::vector<double> Ap(n);

      std::copy(b.begin(), b.end(), r.begin());
      std::copy(r.begin(), r.end(), p.begin());

      double r_sq = 0.0;
      for (int i = 0; i < n; ++i) {
        r_sq += r[i] * r[i];
      }

      int iteration = 0;

      while (iteration < MAX_ITERATIONS && std::sqrt(r_sq) > TOLERANCE) {
        std::fill(Ap.begin(), Ap.end(), 0.0);
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            Ap[i] += A[i][j] * p[j];
          }
        }

        double pAp = 0.0;
        for (int i = 0; i < n; ++i) {
          pAp += p[i] * Ap[i];
        }

        if (std::abs(pAp) < 1e-15) {
          break;
        }

        double alpha = r_sq / pAp;

        for (int i = 0; i < n; ++i) {
          x[i] += alpha * p[i];
          r[i] -= alpha * Ap[i];
        }

        double r_sq_new = 0.0;
        for (int i = 0; i < n; ++i) {
          r_sq_new += r[i] * r[i];
        }

        double beta = r_sq_new / r_sq;

        for (int i = 0; i < n; ++i) {
          p[i] = r[i] + beta * p[i];
        }

        r_sq = r_sq_new;
        iteration++;
      }
    }

    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return x;
  }

  auto range = CalculateRowRange(size, rank, n);
  int start_row = range[0];
  int end_row = range[1];

  if (start_row < 0 || end_row < 0) {
    return std::vector<double>();
  }

  int local_rows = end_row - start_row + 1;

  const double TOLERANCE = 1e-6;
  const int MAX_ITERATIONS = n * 2;

  std::vector<double> local_x(local_rows, 0.0);
  std::vector<double> local_r(local_rows);
  std::vector<double> local_p(local_rows);
  std::vector<double> local_Ap(local_rows);

  for (int i = 0; i < local_rows; ++i) {
    int global_idx = start_row + i;
    local_r[i] = b[global_idx];
    local_p[i] = local_r[i];
  }

  std::vector<double> global_x(n, 0.0);
  std::vector<double> global_p(n);

  if (rank == 0) {
    for (int i = 0; i < local_rows; ++i) {
      global_p[start_row + i] = local_p[i];
    }

    for (int proc = 1; proc < size; ++proc) {
      auto proc_range = CalculateRowRange(size, proc, n);
      int proc_start = proc_range[0];
      int proc_rows = proc_range[1] - proc_start + 1;

      std::vector<double> recv_buf(proc_rows);
      MPI_Recv(recv_buf.data(), proc_rows, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < proc_rows; ++i) {
        global_p[proc_start + i] = recv_buf[i];
      }
    }
  } else {
    MPI_Send(local_p.data(), local_rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(global_p.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double r_sq = 0.0;
  for (int i = 0; i < local_rows; ++i) {
    r_sq += local_r[i] * local_r[i];
  }

  double global_r_sq = 0.0;
  MPI_Allreduce(&r_sq, &global_r_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int iteration = 0;

  while (iteration < MAX_ITERATIONS && std::sqrt(global_r_sq) > TOLERANCE) {
    std::fill(local_Ap.begin(), local_Ap.end(), 0.0);
    for (int i = 0; i < local_rows; ++i) {
      int global_row = start_row + i;
      for (int j = 0; j < n; ++j) {
        local_Ap[i] += A[global_row][j] * global_p[j];
      }
    }

    double pAp_local = 0.0;
    for (int i = 0; i < local_rows; ++i) {
      pAp_local += local_p[i] * local_Ap[i];
    }

    double pAp_global = 0.0;
    MPI_Allreduce(&pAp_local, &pAp_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (std::abs(pAp_global) < 1e-15) {
      break;
    }

    double alpha = global_r_sq / pAp_global;

    for (int i = 0; i < local_rows; ++i) {
      local_x[i] += alpha * local_p[i];
      local_r[i] -= alpha * local_Ap[i];
    }

    double r_sq_new_local = 0.0;
    for (int i = 0; i < local_rows; ++i) {
      r_sq_new_local += local_r[i] * local_r[i];
    }

    double global_r_sq_new = 0.0;
    MPI_Allreduce(&r_sq_new_local, &global_r_sq_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double beta = global_r_sq_new / global_r_sq;

    for (int i = 0; i < local_rows; ++i) {
      local_p[i] = local_r[i] + beta * local_p[i];
    }

    if (rank == 0) {
      for (int i = 0; i < local_rows; ++i) {
        global_p[start_row + i] = local_p[i];
      }

      for (int proc = 1; proc < size; ++proc) {
        auto proc_range = CalculateRowRange(size, proc, n);
        int proc_start = proc_range[0];
        int proc_rows = proc_range[1] - proc_start + 1;

        std::vector<double> recv_buf(proc_rows);
        MPI_Recv(recv_buf.data(), proc_rows, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < proc_rows; ++i) {
          global_p[proc_start + i] = recv_buf[i];
        }
      }
    } else {
      MPI_Send(local_p.data(), local_rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(global_p.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    global_r_sq = global_r_sq_new;
    iteration++;
  }

  if (rank == 0) {
    for (int i = 0; i < local_rows; ++i) {
      global_x[start_row + i] = local_x[i];
    }

    for (int proc = 1; proc < size; ++proc) {
      auto proc_range = CalculateRowRange(size, proc, n);
      int proc_start = proc_range[0];
      int proc_rows = proc_range[1] - proc_start + 1;

      std::vector<double> recv_buf(proc_rows);
      MPI_Recv(recv_buf.data(), proc_rows, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < proc_rows; ++i) {
        global_x[proc_start + i] = recv_buf[i];
      }
    }
  } else {
    MPI_Send(local_x.data(), local_rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

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
    const auto &[A, b] = GetInput();

    if (A.empty() && b.empty()) {
      is_valid = true;
    } else if (A.empty() || b.empty()) {
      is_valid = false;
    } else {
      is_valid = IsSystemCompatible(A, b);
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
  const auto &[A, b] = input_data;

  if (A.empty() && b.empty()) {
    GetOutput() = std::vector<double>();
    return true;
  }

  BroadcastSystemData(rank, input_data);

  std::vector<double> solution = SolveSystemMPI(rank, size, A, b);

  GetOutput() = solution;

  return true;
}

bool RemizovKSystemLinearEquationsGradientMPI::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_max_in_matrix_string
