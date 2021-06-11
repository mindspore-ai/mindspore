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
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (indices_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "SparseToDense requires 'indices' should be a 2-D Tensor, but got " << indices_shape.size()
                      << "-D";
  }
  auto values_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (values_shape.size() != 1 || values_shape[0] != indices_shape[0]) {
    MS_LOG(EXCEPTION)
      << "SparseToDense requires 'values' should be a 1-D Tensor and the first dimension length should be "
         "equal to the 'indices' first dimension length, but got 'values' shape: "
      << values_shape;
  }
  values_size_ = values_shape[0];
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
}

template <typename I, typename T>
bool SparseToDenseCPUKernel<I, T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> & /*workspace*/,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto indices_addr = reinterpret_cast<I *>(inputs[0]->addr);
  auto values_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t output_length = outputs[0]->size / sizeof(T);
  if (memset_s(output_addr, output_length, 0, output_length) != EOK) {
    MS_LOG(EXCEPTION) << "Memset Failed!";
  }

  size_t rank = output_shape_.size();
  for (size_t i = 0; i < values_size_; ++i) {
    size_t out_index = 0;
    for (size_t j = 0; j < rank; j++) {
      int index = indices_addr[i * rank + j];
      if (index >= SizeToInt(output_shape_[j]) || index < 0) {
        MS_EXCEPTION(ValueError) << "The " << i << "th value in " << j << "th dimension index: " << index
                                 << " out of bounds: [0, " << output_shape_[j] << ")";
      }
      size_t count = 1;
      for (size_t k = j + 1; k < rank; k++) {
        count *= output_shape_[k];
      }
      out_index += SizeToInt(index) * count;
    }
    output_addr[out_index] = values_addr[i];
  }
  return true;
}

template <typename I, typename T>
void SparseToDenseCPUKernel<I, T>::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "SparseToDense needs 3 inputs, but got " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "SparseToDense should have 2 outputs, but got " << output_num;
  }
}
}  // namespace kernel
}  // namespace mindspore
