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
#include <cmath>
#include <string>
#include <thread>
#include "backend/kernel_compiler/cpu/elu_grad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void EluGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (dtype_ != AnfAlgo::GetInputDeviceDataType(kernel_node, 1)) {
    MS_LOG(EXCEPTION) << "Input0 and input1 must has the same data type";
  }
}

bool EluGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << "is not support.";
  }
  return true;
}

template <typename T>
void EluGradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                    const std::vector<AddressPtr> &outputs) const {
  T *input0 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input1 = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto task = [input0, input1, output](const size_t start, const size_t end) {
    const T alpha = T(1);
    for (size_t i = start; i < end; i++) {
      output[i] = (input1[i] < static_cast<T>(0)) ? input0[i] * (input1[i] + alpha) : input0[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, lens);
}
}  // namespace kernel
}  // namespace mindspore
