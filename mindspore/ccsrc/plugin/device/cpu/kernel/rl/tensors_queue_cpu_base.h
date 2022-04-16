/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSORS_QUEUE_BASE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSORS_QUEUE_BASE_H_

#include <vector>
#include <mutex>
#include <condition_variable>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorsQueueMgr;
using mindspore::device::cpu::CPUTensorsQueue;
using mindspore::device::cpu::CPUTensorsQueuePtr;

class TensorsQueueCPUBaseMod : public DeprecatedNativeCpuKernelMod {
 public:
  TensorsQueueCPUBaseMod() = default;
  ~TensorsQueueCPUBaseMod() = default;

  inline CPUTensorsQueuePtr GetTensorsQueue(const std::vector<AddressPtr> &inputs) {
    auto handle = GetDeviceAddress<int64_t>(inputs, 0);
    MS_EXCEPTION_IF_NULL(handle);
    auto tensors_q =
      std::dynamic_pointer_cast<CPUTensorsQueue>(TensorsQueueMgr::GetInstance().GetTensorsQueue(handle[0]));
    MS_EXCEPTION_IF_NULL(tensors_q);
    return tensors_q;
  }

 protected:
  // Lock the operation: Get, Pop, Size and Clear.
  static std::mutex tq_mutex_;
  static std::condition_variable read_cdv_;
  static std::condition_variable write_cdv_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSORS_QUEUE_BASE_H_
