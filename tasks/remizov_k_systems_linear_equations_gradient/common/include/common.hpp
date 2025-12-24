#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace remizov_k_systems_linear_equations_gradient {

using InType = std::tuple<std::vector<std::vector<double>>, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<std::tuple<std::vector<std::vector<double>>, std::vector<double>>, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace remizov_k_systems_linear_equations_gradient
