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

#include "backend/kernel_compiler/cpu/eigen/lu_cpu_kernel.h"
#include <vector>
#include "utils/ms_utils.h"
#include "Eigen/Dense"
#include "Eigen/LU"

namespace mindspore {
namespace kernel {

namespace {
constexpr size_t kLUInputsNum = 1;
constexpr size_t kLUaIndex = 0;
constexpr size_t kLUOutputsNum = 3;
constexpr size_t kLuIndex = 0;
constexpr size_t kPivotsIndex = 1;
constexpr size_t kPermutationIndex = 2;
constexpr size_t kLUDefaultShape = 1;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
}  // namespace

template <typename T>
void LUCPUKernel<T>::InitMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col) {
  if (shape.empty()) {
    MS_LOG_EXCEPTION << kernel_name_ << "shape is invalid.";
  }
  if (shape.size() == kLUDefaultShape) {
    *row = shape.front();
    *col = 1;
  } else {
    *row = shape.at(shape.size() - kRowIndex);
    *col = shape.at(shape.size() - kColIndex);
  }
  return;
}

template <typename T>
void LUCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kLUInputsNum, kernel_name_);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kLUOutputsNum, kernel_name_);
  auto a_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kLUaIndex);
  InitMatrixInfo(a_shape, &a_row_, &a_col_);
  auto lu_shape = AnfAlgo::GetOutputInferShape(kernel_node, kLuIndex);
  InitMatrixInfo(lu_shape, &lu_row_, &lu_col_);
  auto pivots_shape = AnfAlgo::GetOutputInferShape(kernel_node, kPivotsIndex);
  InitMatrixInfo(pivots_shape, &pivots_row_, &pivots_col_);
  auto permutation_shape = AnfAlgo::GetOutputInferShape(kernel_node, kPermutationIndex);
  InitMatrixInfo(permutation_shape, &permutation_row_, &permutation_col_);
}

template <typename T>
bool LUCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                            const std::vector<kernel::AddressPtr> &outputs) {
  T *a_value = reinterpret_cast<T *>(inputs[kLUaIndex]->addr);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_a(a_value, a_row_, a_col_);

  T *lu_value = reinterpret_cast<T *>(outputs[kLuIndex]->addr);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_lu(lu_value, lu_row_, lu_col_);
  int *pivots_value = reinterpret_cast<int *>(outputs[kPivotsIndex]->addr);
  Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_pivots(
    pivots_value, pivots_row_, pivots_col_);
  int *permutation_value = reinterpret_cast<int *>(outputs[kPermutationIndex]->addr);
  Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_permutation(
    permutation_value, permutation_row_, permutation_col_);

  if (a_row_ == a_col_) {
    // partial_piv_lu
    output_lu = input_a.lu().matrixLU();
    output_pivots = input_a.lu().permutationP().indices();
  } else {
    // full_piv_lu
    output_lu = input_a.fullPivLu().matrixLU();
    output_pivots = input_a.fullPivLu().permutationP().indices();
  }
  output_permutation = output_pivots;
  if (output_lu.RowsAtCompileTime != 0 && output_lu.ColsAtCompileTime != 0 && output_permutation.size() != 0) {
    return true;
  }
  MS_LOG_EXCEPTION << kernel_name_ << " output lu shape invalid.";
}
}  // namespace kernel
}  // namespace mindspore
