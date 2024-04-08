/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/pyboost/pyboost_gpu_kernel_extra_func.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/hal/profiler/gpu_profiling.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "kernel/pyboost/pyboost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
bool PyboostGPUKernelExtraFunc::IsKernelModRegistered(const std::string &op_name) {
  return kernel::Factory<kernel::NativeGpuKernelMod>::Instance().IsRegistered(op_name);
}

bool PyboostGPUKernelExtraFunc::IsEnableProfiler() {
  const auto &profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  return profiler_inst->GetEnableFlag() && profiler_inst->GetOpTimeFlag();
}

void PyboostGPUKernelExtraFunc::LaunchKernelWithProfiler(const std::string &op_name,
                                                         const device::DeviceContext *device_context,
                                                         const std::vector<BaseShapePtr> &base_shape,
                                                         const std::function<void()> &func) {
  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  profiler_inst->OpDataProducerBegin(op_name, device::gpu::GPUDeviceManager::GetInstance().default_stream());
  func();
  profiler_inst->OpDataProducerEnd();
  profiler_inst->RecordFrameWorkInfo(op_name, base_shape);

  if (profiler_inst->GetSyncEnableFlag()) {
    if (!device_context->device_res_manager_->SyncAllStreams()) {
      MS_LOG(EXCEPTION) << "Profiler SyncStream failed.";
    }
  }
}

REG_PYBOOST_KERNEL_EXTRA_FUN(GPU, PyboostGPUKernelExtraFunc);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
