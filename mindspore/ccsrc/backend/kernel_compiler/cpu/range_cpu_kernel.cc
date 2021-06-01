/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/range_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void RangeCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool RangeCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                            const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    return LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Only support int, float, but actual data type is " << TypeIdLabel(dtype_);
  }
}

template <typename T>
bool RangeCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  T start_ = reinterpret_cast<T *>(inputs[0]->addr)[0];
  T limit_ = reinterpret_cast<T *>(inputs[1]->addr)[0];
  T delta_ = reinterpret_cast<T *>(inputs[2]->addr)[0];

  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t elem_num = outputs[0]->size / sizeof(T);
  for (size_t i = 0; i < elem_num; i++) {
    T val_ = start_ + i * delta_;
    if (val_ > limit_) {
      break;
    }
    output_addr[i] = val_;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
