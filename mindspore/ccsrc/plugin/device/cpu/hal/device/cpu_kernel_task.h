/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_TASK_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_TASK_H_
#include <vector>
#include <memory>
#include <utility>
#include "include/backend/device_address.h"
#include "ir/tensor_storage_info.h"
#include "runtime/hardware/device_context.h"
#include "runtime/pynative/async/kernel_task.h"

namespace mindspore::device::cpu {
class CpuContiguousKernelTask : public pynative::KernelTask {
 public:
  explicit CpuContiguousKernelTask(std::shared_ptr<pynative::KernelTaskContext> context)
      : pynative::KernelTask(std::move(context)) {}
  ~CpuContiguousKernelTask() = default;

  bool RunWithRet() override;
};
class CpuCopyWithSliceKernelTask : public pynative::KernelTask {
 public:
  explicit CpuCopyWithSliceKernelTask(std::shared_ptr<pynative::KernelTaskContext> context)
      : pynative::KernelTask(std::move(context)) {}
  ~CpuCopyWithSliceKernelTask() = default;

  bool RunWithRet() override;
};
}  // namespace mindspore::device::cpu

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_TASK_H_
