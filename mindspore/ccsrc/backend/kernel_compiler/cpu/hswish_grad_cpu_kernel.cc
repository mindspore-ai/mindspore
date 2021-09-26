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

#include "backend/kernel_compiler/cpu/hswish_grad_cpu_kernel.h"
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHSwishGradInputsNum = 2;
constexpr size_t kHSwishGradOutputsNum = 1;
}  // namespace

template <typename T>
void HSwishGradCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  for (const uint64_t &d : x_shape_) {
    tensor_size_ *= d;
  }
}

template <typename T>
bool HSwishGradCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHSwishGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHSwishGradOutputsNum, kernel_name_);
  const auto *dy = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *x = reinterpret_cast<T *>(inputs[1]->addr);
  auto *out = reinterpret_cast<T *>(outputs[0]->addr);

  auto zero = static_cast<T>(0);
  auto two = static_cast<T>(2);
  auto three = static_cast<T>(3);
  auto six = static_cast<T>(6);

  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      if (x[i] + three <= zero) {
        out[i] = zero;
      } else if (x[i] >= three) {
        out[i] = dy[i];
      } else {
        out[i] = dy[i] * (two * x[i] + three) / six;
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, tensor_size_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
