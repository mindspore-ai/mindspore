/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/eigen/cholesky_solve_cpu_kernel.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "Eigen/Cholesky"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 2;
constexpr size_t kInputAIndex = 0;
constexpr size_t kInputBIndex = 1;
constexpr size_t kOutputsNum = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
}  // namespace

template <typename T>
void CholeskySolveCpuKernelMod<T>::InitRightMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col) {
  if (shape.empty()) {
    MS_LOG_EXCEPTION << kernel_name_ << " input shape is empty which is invalid.";
  }
  constexpr size_t min_dim = 1;
  if (shape.size() <= min_dim) {
    MS_LOG_EXCEPTION << kernel_name_ << " input shape dim is " << shape.size() << " which is invalid.";
  }
  *row = shape.at(shape.size() - kRowIndex);
  *col = shape.at(shape.size() - kColIndex);
  outer_batch_ = min_dim;
  for (const auto &sh : shape) {
    outer_batch_ *= sh;
  }
  outer_batch_ /= ((*row) * (*col));
}

template <typename T>
void CholeskySolveCpuKernelMod<T>::InitLeftMatrixInfo(const std::vector<size_t> &shape, const bool is_rank_equal,
                                                      size_t *row, size_t *col) {
  if (shape.empty()) {
    MS_LOG_EXCEPTION << kernel_name_ << " input or output shape is empty which is invalid.";
  }
  if (is_rank_equal) {
    *row = shape.at(shape.size() - kRowIndex);
    *col = shape.at(shape.size() - kColIndex);
  } else {
    *row = shape.back();
    *col = 1;
  }
}

template <typename T>
void CholeskySolveCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputsNum, kernel_name_);
  auto input_a_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputAIndex);
  InitRightMatrixInfo(input_a_shape, &input_a_row_, &input_a_col_);
  auto input_b_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputBIndex);
  const bool is_right_equal_left = input_a_shape.size() == input_b_shape.size();
  InitLeftMatrixInfo(input_b_shape, is_right_equal_left, &input_b_row_, &input_b_col_);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, kOutputIndex);
  InitLeftMatrixInfo(output_shape, is_right_equal_left, &output_row_, &output_col_);
  if (common::AnfAlgo::HasNodeAttr(LOWER, kernel_node)) {
    lower_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, LOWER);
  }
  if (input_a_row_ != input_a_col_ || input_a_row_ != input_b_row_) {
    MS_LOG_EXCEPTION << kernel_name_ << " llt solve input a row is not match to b row: " << input_a_row_ << " vs "
                     << input_b_row_;
  }
}

template <typename T>
bool CholeskySolveCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                          const std::vector<AddressPtr> &outputs) {
  T *batch_input_value = reinterpret_cast<T *>(inputs[kInputAIndex]->addr);
  T *batch_input_b_value = reinterpret_cast<T *>(inputs[kInputBIndex]->addr);
  T *batch_output_value = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);
  for (size_t batch = 0; batch < outer_batch_; ++batch) {
    T *input_value = batch_input_value + batch * input_a_row_ * input_a_col_;
    Map<Matrix<T, RowMajor>> input(input_value, input_a_row_, input_a_col_);
    T *input_b_value = batch_input_b_value + batch * input_b_row_ * input_b_row_;
    Map<Matrix<T, RowMajor>> input_b(input_b_value, input_b_row_, input_b_col_);
    T *output_value = batch_output_value + batch * output_row_ * output_col_;
    Map<Matrix<T, RowMajor>> output(output_value, output_row_, output_col_);
    if (lower_) {
      output.noalias() = input.template triangularView<Lower>().solve(input_b);
      input.adjoint().template triangularView<Upper>().solveInPlace(output);
    } else {
      output.noalias() = input.adjoint().template triangularView<Lower>().solve(input_b);
      input.template triangularView<Upper>().solveInPlace(output);
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
