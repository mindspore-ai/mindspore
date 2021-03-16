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

#include "runtime/device/gpu/gpu_launch_kernel.h"

#include <vector>
#include <memory>
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "runtime/device/gpu/gpu_device_manager.h"
#include "runtime/device/gpu/kernel_info_setter.h"
#include "runtime/device/gpu/gpu_kernel_build.h"
#include "abstract/utils.h"

namespace {
constexpr size_t kCommunicationMemAlignSize = 16;
size_t AlignMemorySize(size_t size) {
  if (size == 0) {
    return kCommunicationMemAlignSize;
  }
  return ((size + kCommunicationMemAlignSize - 1) / kCommunicationMemAlignSize) * kCommunicationMemAlignSize;
}
}  // namespace

namespace mindspore::device::gpu {
void GPULaunchkernel::FreeDeviceMem(void *addr) { GPUMemoryAllocator::GetInstance().FreeTensorMem(addr); }

size_t GPULaunchkernel::AlignSizeForLaunchKernel(size_t size) { return AlignMemorySize(size); }

uint8_t *GPULaunchkernel::AllocDeviceMem(size_t size) {
  auto device_memory = GPUMemoryAllocator::GetInstance().AllocTensorMem(size);
  MS_EXCEPTION_IF_NULL(device_memory);
  return static_cast<uint8_t *>(device_memory);
}

void GPULaunchkernel::KernelSelect(std::shared_ptr<session::KernelGraph> kernel_graph) {
  auto node_list = kernel_graph->execution_order();
  for (size_t i = 0; i < node_list.size(); ++i) {
    device::gpu::SetKernelInfo(node_list[i]);
  }
}

void GPULaunchkernel::KernelBuild(std::shared_ptr<session::KernelGraph> kernel_graph) {
  device::gpu::GpuBuild(kernel_graph);
}
}  // namespace mindspore::device::gpu
