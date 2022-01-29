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

#include "backend/kernel_compiler/cpu/eigen/cholesky_cpu_kernel.h"
#include <vector>
#include "backend/kernel_compiler/cpu/eigen/eigen_common_utils.h"
#include "utils/ms_utils.h"
#include "Eigen/Cholesky"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputsNum = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
}  // namespace

template <typename T>
void CholeskyCpuKernelMod<T>::InitMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col) {
  if (shape.empty()) {
    MS_LOG_EXCEPTION << kernel_name_ << " input or output shape is empty which is invalid.";
  }
  constexpr size_t min_dim = 1;
  if (shape.size() <= min_dim) {
    MS_LOG_EXCEPTION << kernel_name_ << " input or output shape dim is " << shape.size() << " which is invalid.";
  }
  *row = shape.at(shape.size() - kRowIndex);
  *col = shape.at(shape.size() - kColIndex);
  outer_batch_ = min_dim;
  for (int batch = 0; batch < static_cast<int>(shape.size() - kRowIndex); ++batch) {
    outer_batch_ *= shape.at(batch);
  }
  if (*row != *col) {
    MS_LOG_EXCEPTION << kernel_name_ << " input shape is invalid. "
                     << "Cholesky expects a square matrix. but input or output shape is: " << *row << ", " << *col;
  }
}

template <typename T>
void CholeskyCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputsNum, kernel_name_);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputsNum, kernel_name_);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputIndex);
  InitMatrixInfo(input_shape, &input_row_, &input_col_);
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, kOutputIndex);
  InitMatrixInfo(output_shape, &output_row_, &output_col_);
  // if clean attribute exits, we will remain rand triangular data by clean flag, otherwise clean it to zero.
  if (AnfAlgo::HasNodeAttr(CLEAN, kernel_node)) {
    clean_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, CLEAN);
  }
  if (AnfAlgo::HasNodeAttr(LOWER, kernel_node)) {
    lower_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, LOWER);
  }
}

template <typename T>
bool CholeskyCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  T *batch_input_value = reinterpret_cast<T *>(inputs[kInputIndex]->addr);
  T *batch_output_value = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);
  Eigen::LLT<Matrix<T, RowMajor>> llt;
  for (size_t batch = 0; batch < outer_batch_; ++batch) {
    T *input_value = batch_input_value + batch * input_row_ * input_col_;
    T *output_value = batch_output_value + batch * output_row_ * output_col_;
    Map<Matrix<T, RowMajor>> input(input_value, input_row_, input_col_);
    Map<Matrix<T, RowMajor>> output(output_value, output_row_, output_col_);
    (void)llt.compute(input);
    if (clean_) {
      if (lower_) {
        output = llt.matrixL();
      } else {
        output = llt.matrixU();
      }
    } else {
      if (lower_) {
        output = llt.matrixLLT();
      } else {
        output = llt.matrixLLT().transpose();
      }
    }
    if (output.RowsAtCompileTime != 0 && output.ColsAtCompileTime != 0) {
      continue;
    } else {
      MS_LOG_EXCEPTION << kernel_name_ << " cholesky llt calculating failed.";
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
