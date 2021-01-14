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

#include "backend/kernel_compiler/cpu/hswish_cpu_kernel.h"
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void HSwishCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo ::GetPrevNodeOutputDeviceDataType(kernel_node, 0);
  if (dtype_ == kTypeUnknown) {
    dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  }
  for (const uint64_t &d : x_shape_) {
    tensor_size_ *= d;
  }

  launch_map_[kNumberTypeInt8] = &HSwishCPUKernel::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &HSwishCPUKernel::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &HSwishCPUKernel::LaunchKernel<int>;
  launch_map_[kNumberTypeInt64] = &HSwishCPUKernel::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeFloat32] = &HSwishCPUKernel::LaunchKernel<float>;

  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Input data type: " << dtype_ << "is not supported for HSwish kernel on CPU.";
  }
}

bool HSwishCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  launch_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void HSwishCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  auto y = reinterpret_cast<T *>(outputs[0]->addr);
  for (uint64_t i = 0; i < tensor_size_; ++i) {
    if (x[i] <= -3) {
      y[i] = 0;
    } else if (x[i] >= 3) {
      y[i] = x[i];
    } else {
      y[i] = x[i] * (x[i] + 3) / 6;
    }
  }
}

void HSwishCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but HSwishCPUKernel needs 1 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but HSwishCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
