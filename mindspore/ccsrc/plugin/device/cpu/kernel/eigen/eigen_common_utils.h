/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIGEN_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIGEN_COMMON_UTILS_H_
#include "Eigen/Dense"
#include "Eigen/Core"
namespace mindspore {
namespace kernel {
using Eigen::ColMajor;
using Eigen::Dynamic;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::RowMajor;
using Eigen::UnitLower;
using Eigen::UnitUpper;
using Eigen::Upper;
template <typename T, int Major>
using Matrix = Eigen::Matrix<T, Dynamic, Dynamic, Major>;
template <typename T>
using MatrixSquare = Eigen::Matrix<T, Dynamic, Dynamic, RowMajor>;
template <typename T>
using ComplexMatrixSquare = Eigen::Matrix<std::complex<T>, Dynamic, Dynamic, RowMajor>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIGEN_COMMON_UTILS_H_
