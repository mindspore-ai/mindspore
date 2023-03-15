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
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace device {
namespace gpu {
const size_t kGBToByte = 1024 << 20;
constexpr float kReservedMemoryRatio = 0.0625;  // 1/16
static const size_t MEM_ALIGN_SIZE = 512;

bool GPUMemoryAllocator::Init() {
  size_t total_size = CudaDriver::total_mem_size();
  size_t free_size = CudaDriver::free_mem_size();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  limited_device_memory_ = context_ptr->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY);
  available_device_memory_ = FloatToSize(limited_device_memory_ * kGBToByte);
  if (total_size > 0 && free_size > 0 && available_device_memory_ > 0) {
    MS_LOG(INFO) << "GPU device total memory size " << total_size << ", current free memory size " << free_size
                 << ", set max available memory size " << available_device_memory_ << ".";
  } else {
    MS_LOG(EXCEPTION) << "#umsg#GPU memory error:#umsg#The total size or free size or max_device_memory size of GPU "
                         "memory can't be zero, total memory size "
                      << total_size << ", current free memory size " << free_size << ", set max available memory size "
                      << available_device_memory_ << ".";
  }
  // In gpu mode, recommend 1/16 reserved for other cuda functions
  if (available_device_memory_ > total_size) {
    size_t recommend_mem_size_for_others = FloatToSize(total_size * kReservedMemoryRatio);
    SetMemPoolBlockSize(std::min(available_device_memory_, total_size - recommend_mem_size_for_others));
  } else {
    SetMemPoolBlockSize(std::min(available_device_memory_, total_size));
  }
  return true;
}

void GPUMemoryAllocator::CheckMaxDeviceMemory() const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto max_device_memory = context_ptr->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY);
  //  Currently not support modifying the max device memory.
  if (!common::IsFloatEqual(limited_device_memory_, max_device_memory)) {
    MS_LOG(EXCEPTION) << "#umsg#Can't change or set context param max_device_memory during running:#umsg#Currently "
                         "effective max_device_memory("
                      << limited_device_memory_ << "GB), set new max_device_memory(" << max_device_memory
                      << "GB) failed.";
  }
}

bool GPUMemoryAllocator::AllocBufferQueueMem(size_t size, DeviceMemPtr *addr) {
  auto alloc_size = AllocDeviceMem(size, addr);
  buffer_q_addr_ = *addr;
  // Buffer queue needs to ensure that the alloc_size and size is equal.
  return alloc_size == size;
}

size_t GPUMemoryAllocator::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  if (size == 0) {
    MS_LOG(EXCEPTION) << "#umsg#GPU memory error:#umsg#The memory alloc size is 0.";
  }
  auto free_size = free_mem_size();
  if (size > free_size) {
    MS_LOG(EXCEPTION) << "#umsg#Memory not enough:#umsg#Current free memory size[" << free_size
                      << "] is smaller than required size[" << size << "].";
  }

  auto alloc_size = CudaDriver::AllocDeviceMem(size, addr);
  if (alloc_size == 0) {
    MS_LOG(EXCEPTION) << "#umsg#Memory not enough:#umsg#Alloc device memory[" << size << "] failed.";
  }
  total_used_device_memory_ += alloc_size;
  available_device_memory_ -= alloc_size;
  MS_LOG(INFO) << "Cuda current free memory size[" << free_size << "], alloc size[" << alloc_size
               << "], left free memory size[" << free_size - alloc_size << "]"
               << ".Total used size[" << total_used_device_memory_ << "].";
  return alloc_size;
}

bool GPUMemoryAllocator::FreeDeviceMem(const DeviceMemPtr &addr) { return CudaDriver::FreeDeviceMem(addr); }

size_t GPUMemoryAllocator::free_mem_size() { return std::min(CudaDriver::free_mem_size(), available_device_memory_); }

size_t GPUMemoryAllocator::AlignMemorySize(size_t size) const {
  if (size == 0) {
    return MEM_ALIGN_SIZE;
  }
  return ((size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE) * MEM_ALIGN_SIZE;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
