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

#include "backend/kernel_compiler/cpu/sparse_to_dense_cpu_kernal.h"
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename I, typename T>
void SparseToDenseCPUKernel<I, T>::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  indices_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  values_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  dense_shape_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (!indices_shape_.size() || !values_shape_.size() || !output_shape_.size()) {
    MS_LOG(EXCEPTION) << "Input NULL";
  }
  if (indices_shape_.size() > 2 || indices_shape_[0] != values_shape_[0]) {
    MS_LOG(EXCEPTION) << "Input Error";
  }
}

size_t DenseGetTensorLen(const std::vector<size_t> &shape) {
  size_t len = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    len *= shape[i];
  }
  return len;
}

template <typename I, typename T>
bool SparseToDenseCPUKernel<I, T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> & /*workspace*/,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto indices_addr = reinterpret_cast<I *>(inputs[0]->addr);
  auto values_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  size_t output_len = DenseGetTensorLen(output_shape_);
  memset(output_addr, 0, output_len * sizeof(T));
  std::vector<size_t> cargo(output_shape_.size(), 0);

  size_t i = output_shape_.size() - 1;
  switch (indices_shape_.size()) {
    case 1:
      for (i = 0; i < indices_shape_[0]; i++) {
        output_addr[indices_addr[i]] = values_addr[i];
      }
      break;

    case 2:
      cargo[i] = 1;
      for (; i >= 1; i--) {
        cargo[i - 1] = cargo[i] * output_shape_[i];
      }
      for (i = 0; i < indices_shape_[0]; i++) {
        size_t out_index = 0;
        for (size_t j = 0; j < indices_shape_[1]; j++) {
          out_index += (*(indices_addr + i * indices_shape_[1] + j)) * cargo[j];
        }
        output_addr[out_index] = values_addr[i];
      }
      break;

    default:
      break;
  }
  return true;
}

template <typename I, typename T>
void SparseToDenseCPUKernel<I, T>::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but SparseToDenseCPUKernel needs 3 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but SparseToDenseCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
