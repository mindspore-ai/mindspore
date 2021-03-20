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

#include "backend/kernel_compiler/cpu/broadcast_to_cpu_kernel.h"

namespace mindspore {
namespace kernel {

template <typename T>
void BroadcastToCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);

  size_t offset = output_shape_.size() - input_shape_.size();
  for (size_t i = 0; i < offset; ++i) {
    input_shape_.insert(input_shape_.begin(), 1);
  }

  for (size_t i = 0; i < input_shape_.size(); ++i) {
    if (output_shape_[i] < input_shape_[i] || output_shape_[i] % input_shape_[i] != 0) {
      MS_LOG(EXCEPTION) << "Cannot broadcast input tensor with shape " << input_shape_ << " to "
                        << "output tensor with shape " << output_shape_
                        << ". Output shape must be the integer times of input shape at the " << i << " dim!";
    }
  }
  for (size_t j = 0; j < output_shape_.size(); j++) {
    nums_ *= output_shape_[j];
  }

  tmp_ptr_ = reinterpret_cast<T *>(malloc(nums_ * sizeof(T)));
}

// BroadcastTo
template <typename T>
void BroadcastToCPUKernel<T>::BroadcastToImpl(size_t dim) {
  if (dim == output_shape_.size() - 1) {
    size_t input_nums = 1;
    for (size_t j = 0; j < input_shape_.size() - 1; ++j) {
      input_nums *= input_shape_[j];
    }
    size_t rate = output_shape_[dim] / input_shape_[dim];

    for (size_t j = 0; j < input_nums; ++j) {
      T *in_ptr = input_ptr_ + input_shape_[dim] * j;
      for (size_t i = 0; i < rate; ++i) {
        T *out_ptr = tmp_ptr_ + (j * rate + i) * input_shape_[dim];
        memcpy_s(out_ptr, input_shape_[dim] * sizeof(T), in_ptr, input_shape_[dim] * sizeof(T));
      }
    }
    size_t elems = input_shape_[dim] * rate * input_nums;
    memcpy_s(output_ptr_, elems * sizeof(T), tmp_ptr_, elems * sizeof(T));
    return;
  }

  BroadcastToImpl(dim + 1);

  size_t rate = output_shape_[dim] / input_shape_[dim];
  if (rate > 1) {
    size_t elems_nums = 1;
    for (size_t j = output_shape_.size() - 1; j > dim; --j) {
      elems_nums *= output_shape_[j];
    }
    size_t input_nums = 1;
    for (size_t j = 0; j < dim; ++j) {
      input_nums *= input_shape_[j];
    }

    for (size_t j = 0; j < input_nums; ++j) {
      T *in_ptr = output_ptr_ + elems_nums * j;
      for (size_t i = 0; i < rate; ++i) {
        T *out_ptr = tmp_ptr_ + (j * rate + i) * elems_nums;
        memcpy_s(out_ptr, elems_nums * sizeof(T), in_ptr, elems_nums * sizeof(T));
      }
    }
    size_t elems = elems_nums * rate * input_nums;
    memcpy_s(output_ptr_, elems * sizeof(T), tmp_ptr_, elems * sizeof(T));
  }
}

template <typename T>
bool BroadcastToCPUKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "Wrong number of inputs or outputs!";
    return false;
  }

  if ((inputs[0] == nullptr) || (inputs[0]->size == 0)) {
    MS_LOG(EXCEPTION) << "Input data is NULL!";
    return false;
  }

  if ((outputs[0] == nullptr) || (outputs[0]->size == 0)) {
    MS_LOG(EXCEPTION) << "Output data is NULL!";
    return false;
  }

  input_ptr_ = reinterpret_cast<T *>(inputs[0]->addr);
  output_ptr_ = reinterpret_cast<T *>(outputs[0]->addr);

  BroadcastToImpl(0);

  return true;
}

}  // namespace kernel
}  // namespace mindspore
