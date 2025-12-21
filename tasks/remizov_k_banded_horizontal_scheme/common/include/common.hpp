#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace remizov_k_banded_horizontal_scheme {

using Matrix = std::vector<std::vector<int>>;
using InType = std::tuple<Matrix, Matrix>;
using OutType = Matrix;
using TestType = std::tuple<std::string, std::tuple<Matrix, Matrix>, Matrix>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace remizov_k_banded_horizontal_scheme
