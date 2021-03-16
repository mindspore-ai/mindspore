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

#include "runtime/device/gpu/gpu_launch_mul.h"

#include <vector>
#include <memory>
#include "abstract/utils.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "runtime/device/gpu/gpu_device_manager.h"
#include "backend/session/single_kernel_graph.h"
#include "frontend/parallel/context.h"

namespace mindspore::device::gpu {
void GPULaunchMul::FreeDeviceMem(void *addr) { GPULaunchkernel::FreeDeviceMem(addr); }

size_t GPULaunchMul::AlignSizeForLaunchKernel(size_t size) { return GPULaunchkernel::AlignSizeForLaunchKernel(size); }

uint8_t *GPULaunchMul::AllocDeviceMem(size_t size) { return GPULaunchkernel::AllocDeviceMem(size); }

void GPULaunchMul::KernelSelect(std::shared_ptr<session::KernelGraph> kernel_graph) {
  GPULaunchkernel::KernelSelect(kernel_graph);
}

void GPULaunchMul::KernelBuild(std::shared_ptr<session::KernelGraph> kernel_graph) {
  GPULaunchkernel::KernelBuild(kernel_graph);
}

void GPULaunchMul::LaunchOpKernel() {
  kernel_mod_ = ObtainLaunchMulKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  // construct mul inputs addr
  ObtainMulInputsAddr();
  // launch mul
  LaunchSingleKernel(inputs_addr_);
}

void GPULaunchMul::FreeLaunchDeviceMem() {
  FreeInputDeviceMemory();
  FreeOutputAndWorkspaceDeviceMem();
}

void GPULaunchMul::CopyHostMemToDevice(size_t origin_size, size_t) {
  if (!GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(input2_addr_, &input2_value_, origin_size, stream_)) {
    MS_LOG(EXCEPTION) << "Copy memory failed";
  }
}
}  // namespace mindspore::device::gpu
