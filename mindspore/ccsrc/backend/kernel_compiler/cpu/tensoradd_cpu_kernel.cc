/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/tensoradd_cpu_kernel.h"
#include <functional>
#include <vector>

namespace mindspore {
namespace kernel {
template <typename T>
void TensorAddCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  // Init shape ans strides
  input_shape_a_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_shape_b_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
}

template <typename T>
bool TensorAddCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> & /*workspace*/,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr_a = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_addr_b = reinterpret_cast<T *>(inputs[1]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(T);
  if (input_shape_a_ == input_shape_b_) {
    auto task = [output_addr, input_addr_a, input_addr_b](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        output_addr[i] = input_addr_a[i] + input_addr_b[i];
      }
    };
    CPUKernelUtils::ParallelFor(task, output_size);
  } else {  // Broadcast
    BroadcastIterator base_iter(input_shape_a_, input_shape_b_, output_shape_);
    auto task = [&base_iter, output_addr, input_addr_a, input_addr_b](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; ++i) {
        output_addr[i] = input_addr_a[iter.GetInputPosA()] + input_addr_b[iter.GetInputPosB()];
        iter.GenNextPos();
      }
    };
    CPUKernelUtils::ParallelFor(task, output_size);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
