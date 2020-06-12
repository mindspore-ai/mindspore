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

#include <algorithm>
#include "device/gpu/gpu_memory_allocator.h"
#include "device/gpu/cuda_driver.h"
#include "utils/log_adapter.h"
#include "utils/context/ms_context.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace device {
namespace gpu {
bool GPUMemoryAllocator::Init() {
  size_t total_size = total_mem_size();
  size_t free_size = CudaDriver::free_mem_size();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  float max_device_memory = context_ptr->max_device_memory();
  max_available_device_memory_ = FloatToSize(max_device_memory * 1024 * 1024 * 1024);
  if (total_size > 0 && free_size > 0 && max_available_device_memory_ > 0) {
    MS_LOG(INFO) << "GPU device total memory size " << total_size << ", current free memory size " << free_size
                 << ", set max available memory size " << max_available_device_memory_;
  } else {
    MS_LOG(EXCEPTION) << "GPU device memory error, total memory size " << total_size << ", current free memory size "
                      << free_size << ", set max available memory size " << max_available_device_memory_;
  }
  return true;
}

bool GPUMemoryAllocator::Finalize() {
  if (buffer_q_addr_ != nullptr) {
    if (!CudaDriver::FreeDeviceMem(buffer_q_addr_)) {
      MS_LOG(ERROR) << "Could not free buffer queue memory.";
      return false;
    }
  }
  return true;
}

bool GPUMemoryAllocator::AllocBufferQueueMem(size_t size, DeviceMemPtr *addr) {
  auto alloc_size = AllocDeviceMem(size, addr);
  buffer_q_addr_ = *addr;
  // Buffer queue needs to ensure that the alloc_size and size is equal.
  return (alloc_size == size) ? true : false;
}

size_t GPUMemoryAllocator::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  if (size == 0) {
    MS_LOG(EXCEPTION) << "The memory alloc size is 0.";
  }
  auto free_size = free_mem_size();
  if (size > free_size) {
    MS_LOG(EXCEPTION) << "Memory not enough: current free memory size[" << free_size
                      << "] is smaller than required size[" << size << "].";
  }

  auto alloc_size = CudaDriver::AllocDeviceMem(size, addr);
  if (alloc_size == 0) {
    MS_LOG(EXCEPTION) << "Alloc device memory[" << size << "] failed.";
  }
  total_used_device_memory_ += alloc_size;
  max_available_device_memory_ -= alloc_size;
  MS_LOG(INFO) << "Current free memory size[" << free_size - alloc_size << "], current alloc size[" << alloc_size
               << "], total used size[" << total_used_device_memory_ << "].";
  return alloc_size;
}

bool GPUMemoryAllocator::FreeDeviceMem(const DeviceMemPtr &addr) { return CudaDriver::FreeDeviceMem(addr); }

size_t GPUMemoryAllocator::free_mem_size() {
  return std::min(CudaDriver::free_mem_size(), max_available_device_memory_);
}

size_t GPUMemoryAllocator::total_mem_size() { return CudaDriver::total_mem_size(); }
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
