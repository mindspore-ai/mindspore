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

#include "backend/kernel_compiler/cpu/eigen/matrix_inverse_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/eigen/eigen_common_utils.h"
#include "Eigen/Dense"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatrixInverseInputsNum = 1;
constexpr size_t kMatrixInverseOutputsNum = 1;
constexpr size_t kMatrixInverseInIndex = 0;
constexpr size_t kMatrixInverseOutIndex = 0;
}  // namespace

template <typename T>
void MatrixInverseCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kMatrixInverseInputsNum, kernel_name_);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kMatrixInverseOutputsNum, kernel_name_);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kMatrixInverseInIndex);
  if (input_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "The dim entered needs to be greater than 2, but " << input_shape.size() << " was taken";
  }
  size_t last_index = input_shape.size() - 1;
  if (input_shape[last_index] != input_shape[last_index - 1]) {
    MS_LOG(EXCEPTION) << "The last two dimensions of the input matrix should be equal!";
  }
  size_ = input_shape[last_index];
  for (size_t i = 0; i < last_index - 1; i++) {
    batch_size_ *= input_shape[i];
  }
  adjoint_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, ADJOINT);
}

template <typename T>
bool MatrixInverseCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<T *>(inputs[kMatrixInverseInIndex]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[kMatrixInverseOutIndex]->addr);

  for (size_t i = 0; i < batch_size_; i++) {
    size_t offset = i * size_ * size_;
    Map<Matrix<T, RowMajor>> input(input_addr + offset, size_, size_);
    Map<Matrix<T, RowMajor>> output(output_addr + offset, size_, size_);
    output = input.inverse();
    if (output.RowsAtCompileTime == 0 || output.ColsAtCompileTime == 0) {
      MS_LOG_EXCEPTION << kernel_name_ << " output shape is invalid.";
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
