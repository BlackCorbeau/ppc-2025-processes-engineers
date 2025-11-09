#pragma once

#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include "task/include/task.hpp"

namespace remizov_k_max_in_matrix_string {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace remizov_k_max_in_matrix_string
