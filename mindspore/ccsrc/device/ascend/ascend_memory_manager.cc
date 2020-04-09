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

#include "device/ascend/ascend_memory_manager.h"
#include "device/ascend/ascend_memory_allocator.h"
#include "utils/context/ms_context.h"
#include "runtime/mem.h"
namespace mindspore {
namespace device {
namespace ascend {
static const uint64_t ASCEND_MEM_SIZE = 20;
static const uint64_t ASCEND_MEM_SIZE_BYTE = (ASCEND_MEM_SIZE << 30);

void AscendMemoryManager::MallocDeviceMemory() {
  device_mem_size_ = ASCEND_MEM_SIZE_BYTE;
  static_mem_offset_ = FloatToSize(device_mem_size_ * GRAPH_INIT_ASCEND_MEM_RATIO);
  auto ret = rtMalloc(reinterpret_cast<void **>(&device_mem_base_), static_mem_offset_, RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << static_mem_offset_ << "] fail, ret[" << ret << "]";
  }
  device_mem_pool_size_ = FloatToSize(device_mem_size_ * (1 - GRAPH_INIT_ASCEND_MEM_RATIO));
  ret = rtMalloc(reinterpret_cast<void **>(&device_mem_pool_base_), device_mem_pool_size_, RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << device_mem_pool_size_ << "] fail, ret[" << ret << "]";
  }
  AscendMemoryAllocator::GetInstance().set_device_mem_pool_base(device_mem_pool_base_);
  AscendMemoryAllocator::GetInstance().set_device_mem_pool_size(device_mem_pool_size_);
}

void AscendMemoryManager::FreeDeviceMemory() {
  if (device_mem_base_ != nullptr) {
    auto ret = rtFree(device_mem_base_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "rtFree mem size[" << device_mem_size_ << "] fail, ret[" << ret << "]";
    }
    device_mem_base_ = nullptr;
  }
  if (device_mem_pool_base_ != nullptr) {
    auto ret = rtFree(device_mem_pool_base_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "rtFree mem size[" << device_mem_pool_size_ << "] fail, ret[" << ret << "]";
    }
    device_mem_pool_base_ = nullptr;
  }
}

void *AscendMemoryManager::AllocTensorMemDynamic(size_t size) {
  return AscendMemoryAllocator::GetInstance().AllocTensorMem(size);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
