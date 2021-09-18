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

#include "backend/kernel_compiler/cpu/print_cpu_kernel.h"
#include <algorithm>
#include "ir/tensor.h"
#include "runtime/device/cpu/cpu_device_address.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace kernel {
template <typename T>
void PrintCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_tensor_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t i = 0; i < input_tensor_num; ++i) {
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
    (void)input_shapes_.emplace_back(input_shape);
    size_t size = input_shape.size() ? 1 : 0;
    for (size_t j = 0; j < input_shape.size(); ++j) {
      size *= input_shape[j];
    }
    (void)input_sizes_.emplace_back(size);
  }
}

template <typename T>
bool PrintCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> & /* workspace */,
                               const std::vector<kernel::AddressPtr> & /* outputs */) {
  auto data_type = CheckType();
  if (data_type == kTypeUnknown) {
    MS_LOG(EXCEPTION) << "CPU print does not support the input type.";
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (input_sizes_[i] == 0) {
      auto num = reinterpret_cast<T *>(inputs[i]->addr);
      std::cout << *num << std::endl;
    } else {
      ShapeVector shape;
      (void)std::transform(input_shapes_[i].begin(), input_shapes_[i].end(), std::back_inserter(shape),
                           [](const size_t &value) { return SizeToLong(value); });
      Tensor tensor(data_type, shape, inputs[i]->addr, input_sizes_[i] * sizeof(T));
      std::cout << tensor.ToStringNoLimit() << std::endl;
    }
  }
  return true;
}

template <typename T>
TypeId PrintCPUKernel<T>::CheckType() {
  if constexpr (std::is_same_v<T, bool>) {
    return kNumberTypeBool;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return kNumberTypeInt8;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return kNumberTypeInt16;
  } else if constexpr (std::is_same_v<T, int>) {
    return kNumberTypeInt32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return kNumberTypeInt64;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return kNumberTypeUInt8;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return kNumberTypeUInt16;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return kNumberTypeUInt32;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return kNumberTypeUInt64;
  } else if constexpr (std::is_same_v<T, float16>) {
    return kNumberTypeFloat16;
  } else if constexpr (std::is_same_v<T, float>) {
    return kNumberTypeFloat32;
  }
  return kTypeUnknown;
}
}  // namespace kernel
}  // namespace mindspore
