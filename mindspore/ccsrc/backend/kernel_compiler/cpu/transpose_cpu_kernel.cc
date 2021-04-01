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

#include "backend/kernel_compiler/cpu/transpose_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void TransposeCPUFwdKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  auto tmp = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "perm");
  axes_ = {tmp.begin(), tmp.end()};
  dtype_ = AnfAlgo ::GetPrevNodeOutputDeviceDataType(kernel_node, 0);
  if (dtype_ == kTypeUnknown) {
    dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  }

  launch_map_[kNumberTypeInt8] = &TransposeCPUFwdKernel::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &TransposeCPUFwdKernel::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &TransposeCPUFwdKernel::LaunchKernel<int>;
  launch_map_[kNumberTypeInt64] = &TransposeCPUFwdKernel::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeUInt8] = &TransposeCPUFwdKernel::LaunchKernel<uint8_t>;
  launch_map_[kNumberTypeUInt16] = &TransposeCPUFwdKernel::LaunchKernel<uint16_t>;
  launch_map_[kNumberTypeUInt32] = &TransposeCPUFwdKernel::LaunchKernel<uint32_t>;
  launch_map_[kNumberTypeUInt64] = &TransposeCPUFwdKernel::LaunchKernel<uint64_t>;
  launch_map_[kNumberTypeFloat32] = &TransposeCPUFwdKernel::LaunchKernel<float>;
  launch_map_[kNumberTypeBool] = &TransposeCPUFwdKernel::LaunchKernel<bool>;

  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Input data type: " << dtype_ << "is not supported for Transpose kernel on CPU.";
  }
}

bool TransposeCPUFwdKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> & /*workspace*/,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  launch_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void TransposeCPUFwdKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t size = IntToSize(inputs[0]->size / sizeof(T));
  TransposeIterator base_iter(output_shape_, axes_, input_shape_);
  auto task = [&base_iter, input_addr, output_addr](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; ++i) {
      output_addr[i] = input_addr[iter.GetPos()];
      iter.GenNextPos();
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}
}  // namespace kernel
}  // namespace mindspore
