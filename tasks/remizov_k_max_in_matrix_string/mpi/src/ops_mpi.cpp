#include "remizov_k_max_in_matrix_string/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "remizov_k_max_in_matrix_string/common/include/common.hpp"

namespace remizov_k_max_in_matrix_string {

RemizovKSystemLinearEquationsGradientMPI::RemizovKSystemLinearEquationsGradientMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool RemizovKSystemLinearEquationsGradientMPI::ValidationImpl() {
  // Валидация выполняется в последовательной части (ранг 0)
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &[A, b] = GetInput();

    if (A.empty() || b.empty()) {
      return false;
    }

    size_t n = A.size();

    if (A[0].size() != n) {
      return false;
    }

    if (b.size() != n) {
      return false;
    }

    // Проверка симметричности (упрощенно)
    for (size_t i = 0; i < n; ++i) {
      if (A[i][i] <= 0) {
        return false;
      }
    }
  }

  // Рассылаем результат валидации всем процессам
  int validation_result = 1;
  if (rank == 0) {
    validation_result = 0;  // По умолчанию валидно
  }

  MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return validation_result == 0;
}

bool RemizovKSystemLinearEquationsGradientMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

double RemizovKSystemLinearEquationsGradientMPI::LocalDotProduct(const std::vector<double> &local_vec,
                                                                 const std::vector<double> &full_vec, int start_row,
                                                                 int end_row) {
  double local_dot = 0.0;

  // Вычисляем локальную часть скалярного произведения
  for (int i = start_row; i <= end_row; ++i) {
    local_dot += local_vec[i - start_row] * full_vec[i];
  }

  // Суммируем результаты со всех процессов
  double global_dot = 0.0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return global_dot;
}

std::vector<double> RemizovKSystemLinearEquationsGradientMPI::LocalMatrixVectorMultiply(
    const std::vector<std::vector<double>> &local_A, const std::vector<double> &x, int n) {
  int start_row = 0, end_row = 0;
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Вычисляем диапазон строк для текущего процесса
  auto ranges = CalculateRowRanges(size, n);
  start_row = ranges[rank].first;
  end_row = ranges[rank].second;

  // Вычисляем локальную часть умножения матрицы на вектор
  std::vector<double> local_result(end_row - start_row + 1, 0.0);

  for (int i = start_row; i <= end_row; ++i) {
    for (int j = 0; j < n; ++j) {
      local_result[i - start_row] += local_A[i - start_row][j] * x[j];
    }
  }

  // Собираем полный результат на всех процессах
  std::vector<double> global_result(n, 0.0);

  // Собираем результаты от всех процессов
  std::vector<int> recv_counts(size);
  std::vector<int> displacements(size);

  for (int i = 0; i < size; ++i) {
    recv_counts[i] = ranges[i].second - ranges[i].first + 1;
    displacements[i] = ranges[i].first;
  }

  MPI_Allgatherv(local_result.data(), end_row - start_row + 1, MPI_DOUBLE, global_result.data(), recv_counts.data(),
                 displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  return global_result;
}

std::vector<std::pair<int, int>> RemizovKSystemLinearEquationsGradientMPI::CalculateRowRanges(int size, int n) {
  std::vector<std::pair<int, int>> ranges(size);

  int base_rows = n / size;
  int extra_rows = n % size;

  int current_start = 0;
  for (int i = 0; i < size; ++i) {
    int rows_for_process = base_rows + (i < extra_rows ? 1 : 0);
    ranges[i] = {current_start, current_start + rows_for_process - 1};
    current_start += rows_for_process;
  }

  return ranges;
}

bool RemizovKSystemLinearEquationsGradientMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[full_A, full_b] = GetInput();
  int n = 0;

  // Процесс 0 определяет размер системы
  if (rank == 0) {
    n = static_cast<int>(full_A.size());

    // Проверка корректности матрицы
    if (n == 0) {
      // Рассылаем информацию о пустой системе
      MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
      GetOutput() = std::vector<double>();
      return true;
    }

    if (full_A[0].size() != static_cast<size_t>(n)) {
      n = -1;  // Ошибка - неквадратная матрица
    }
  }

  // Рассылаем размер системы всем процессам
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0) {
    GetOutput() = std::vector<double>();
    return true;
  }

  // Вычисляем диапазон строк для текущего процесса
  auto ranges = CalculateRowRanges(size, n);
  int start_row = ranges[rank].first;
  int end_row = ranges[rank].second;
  int local_rows = end_row - start_row + 1;

  // Рассылаем вектор b всем процессам
  std::vector<double> b(n);
  if (rank == 0) {
    std::copy(full_b.begin(), full_b.end(), b.begin());
  }
  MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Распределяем матрицу по процессам
  std::vector<std::vector<double>> local_A(local_rows, std::vector<double>(n));

  if (rank == 0) {
    // Процесс 0 копирует свою часть
    for (int i = 0; i < local_rows; ++i) {
      std::copy(full_A[start_row + i].begin(), full_A[start_row + i].end(), local_A[i].begin());
    }

    // Отправляем части другим процессам
    for (int proc = 1; proc < size; ++proc) {
      int proc_start = ranges[proc].first;
      int proc_end = ranges[proc].second;
      int proc_rows = proc_end - proc_start + 1;

      // Отправляем каждую строку отдельно
      for (int row = 0; row < proc_rows; ++row) {
        MPI_Send(full_A[proc_start + row].data(), n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    // Получаем свою часть матрицы
    for (int row = 0; row < local_rows; ++row) {
      MPI_Recv(local_A[row].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Алгоритм сопряженных градиентов
  const int MAX_ITERATIONS = n * 2;  // Максимум 2*n итераций
  const double TOLERANCE = 1e-12;

  std::vector<double> x(n, 0.0);  // Начальное приближение

  // Вычисляем начальную невязку r = b - A*x
  std::vector<double> Ax = LocalMatrixVectorMultiply(local_A, x, n);
  std::vector<double> r(n);
  for (int i = 0; i < n; ++i) {
    r[i] = b[i] - Ax[i];
  }

  std::vector<double> p = r;  // Начальное направление

  // Вычисляем начальную норму невязки
  double r_norm_squared = LocalDotProduct(r, r, start_row, end_row);
  double initial_residual = std::sqrt(r_norm_squared);
  if (initial_residual < TOLERANCE) {
    // Уже сошлось
    if (rank == 0) {
      GetOutput() = x;
    }
    return true;
  }

  int iteration = 0;
  double residual_norm = initial_residual;

  while (iteration < MAX_ITERATIONS && residual_norm > TOLERANCE * initial_residual) {
    // Вычисляем A*p
    std::vector<double> Ap = LocalMatrixVectorMultiply(local_A, p, n);

    // Вычисляем скалярное произведение p*Ap
    double pAp = LocalDotProduct(p, Ap, start_row, end_row);

    if (std::abs(pAp) < 1e-15) {
      break;  // Предотвращаем деление на ноль
    }

    // Вычисляем шаг alpha
    double alpha = r_norm_squared / pAp;

    // Обновляем решение x = x + alpha*p
    for (int i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
    }

    // Обновляем невязку r = r - alpha*Ap
    for (int i = 0; i < n; ++i) {
      r[i] -= alpha * Ap[i];
    }

    // Сохраняем старую норму невязки
    double r_norm_squared_old = r_norm_squared;

    // Вычисляем новую норму невязки
    r_norm_squared = LocalDotProduct(r, r, start_row, end_row);
    residual_norm = std::sqrt(r_norm_squared);

    if (std::abs(r_norm_squared_old) < 1e-15) {
      break;
    }

    // Вычисляем коэффициент beta
    double beta = r_norm_squared / r_norm_squared_old;

    // Обновляем направление p = r + beta*p
    for (int i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    iteration++;
  }

  // Собираем решение на процессе 0
  if (rank == 0) {
    GetOutput() = x;
  }

  // Остальные процессы отправляют свое решение процессу 0
  // (хотя в методе CG все процессы должны иметь одинаковое решение)
  if (rank != 0) {
    MPI_Send(x.data(), n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  } else {
    // Процесс 0 получает решения от других процессов для проверки
    std::vector<double> temp_x(n);
    for (int proc = 1; proc < size; ++proc) {
      MPI_Recv(temp_x.data(), n, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // Можно добавить проверку согласованности решений
    }
  }

  return true;
}

bool RemizovKSystemLinearEquationsGradientMPI::PostProcessingImpl() {
  return true;
}

}  // namespace remizov_k_max_in_matrix_string
