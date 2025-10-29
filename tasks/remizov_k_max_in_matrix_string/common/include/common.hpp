#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace remizov_k_max_in_matrix_string {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace remizov_k_max_in_matrix_string
