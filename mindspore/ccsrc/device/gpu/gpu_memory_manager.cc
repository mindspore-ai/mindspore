/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "device/gpu/gpu_memory_manager.h"
#include "device/gpu/gpu_memory_allocator.h"
#include "utils/context/ms_context.h"
#include "utils/convert_utils.h"
namespace mindspore {
namespace device {
namespace gpu {
void *GPUMemoryManager::MallocMemFromMemPool(size_t size) {
  return GPUMemoryAllocator::GetInstance().AllocTensorMem(size);
}

void GPUMemoryManager::FreeMemFromMemPool(void *device_ptr) {
  GPUMemoryAllocator::GetInstance().FreeTensorMem(device_ptr);
}

std::vector<void *> GPUMemoryManager::MallocContinuousMemFromMemPool(size_t total_size, std::vector<size_t> size_list) {
  return GPUMemoryAllocator::GetInstance().AllocContinuousTensorMem(total_size, size_list);
}

void GPUMemoryManager::MallocDeviceMemory() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // If use the dynamic memory pool, then alloc the first memory block to init.
  if (context_ptr->enable_dynamic_mem_pool()) {
    auto device_addr = MallocMemFromMemPool(1);
    if (!device_addr) {
      MS_LOG(ERROR) << "Dynamic memory pool init error.";
    }
  } else {
    // Need to reserve 20% space for dynamic memory
    const float init_gpu_mem_ratio = 0.8;
    size_t mem_size = FloatToSize(GPUMemoryAllocator::GetInstance().free_mem_size() * init_gpu_mem_ratio);
    auto alloc_size =
      GPUMemoryAllocator::GetInstance().AllocDeviceMem(mem_size, reinterpret_cast<void **>(&device_mem_base_));
    device_mem_size_ = alloc_size;
    static_mem_offset_ = device_mem_size_;
  }
}

void GPUMemoryManager::FreeDeviceMemory() {
  if (device_mem_base_ != nullptr) {
    if (!GPUMemoryAllocator::GetInstance().FreeDeviceMem(device_mem_base_)) {
      MS_LOG(EXCEPTION) << "Could not free gpu device memory.";
    }
  }
  GPUMemoryAllocator::GetInstance().ReleaseDeviceRes();
}

uint8_t *GPUMemoryManager::MallocStaticMem(size_t size, bool) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->enable_dynamic_mem_pool()) {
    auto device_ptr = MallocMemFromMemPool(size);
    MS_EXCEPTION_IF_NULL(device_ptr);
    return AddressOffset(device_ptr, 0);
  }

  auto align_size = GetCommonAlignSize(size);
  if (static_mem_offset_ < align_size) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  auto offset = static_mem_offset_ - align_size;
  if (dynamic_mem_offset_ > offset) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_static_size_ += align_size;
  static_mem_offset_ = offset;
  return device_mem_base_ + offset;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
