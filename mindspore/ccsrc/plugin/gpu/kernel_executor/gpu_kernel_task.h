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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_KERNEL_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_KERNEL_TASK_H_
#include <vector>
#include <memory>
#include <utility>
#include "device_address/device_address.h"
#include "ir/tensor_storage_info.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "runtime/pipeline/task/kernel_task.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore::device::gpu {
class GpuContiguousKernelTask : public runtime::KernelTask {
 public:
  explicit GpuContiguousKernelTask(std::shared_ptr<runtime::KernelTaskContext> context)
      : runtime::KernelTask(std::move(context)) {}
  ~GpuContiguousKernelTask() = default;

  bool RunWithRet() override;
};

class GpuCopyWithSliceKernelTask : public runtime::KernelTask {
 public:
  explicit GpuCopyWithSliceKernelTask(std::shared_ptr<runtime::KernelTaskContext> context)
      : runtime::KernelTask(std::move(context)) {}
  ~GpuCopyWithSliceKernelTask() = default;

  bool RunWithRet() override;
};
}  // namespace mindspore::device::gpu

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_KERNEL_TASK_H_
