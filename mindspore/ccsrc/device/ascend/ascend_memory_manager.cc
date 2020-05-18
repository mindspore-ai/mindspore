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
#include "device/ascend/ascend_memory_pool.h"
#include "utils/context/ms_context.h"
#include "runtime/mem.h"
namespace mindspore {
namespace device {
namespace ascend {
const uint64_t kAscendDeviceMemGB = 26;
const uint64_t kAscendMemPoolGB = 4;
const uint64_t kAscendDeviceMemSize = (kAscendDeviceMemGB << 30);
const uint64_t kAscendMemPoolSize = (kAscendMemPoolGB << 30);

void AscendMemoryManager::MallocDeviceMemory() {
  device_mem_size_ = kAscendDeviceMemSize;
  static_mem_offset_ = device_mem_size_;
  auto ret = rtMalloc(reinterpret_cast<void **>(&device_mem_base_), static_mem_offset_, RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << static_mem_offset_ << "] fail, ret[" << ret << "]";
  }
  device_mem_pool_size_ = kAscendMemPoolSize;
  ret = rtMalloc(reinterpret_cast<void **>(&device_mem_pool_base_), device_mem_pool_size_, RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << device_mem_pool_size_ << "] fail, ret[" << ret << "]";
  }
  AscendMemoryPool::GetInstance().set_device_mem_pool_base(device_mem_pool_base_);
  AscendMemoryPool::GetInstance().set_device_mem_pool_size(device_mem_pool_size_);
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

void *AscendMemoryManager::MallocMemFromMemPool(size_t size) {
  return AscendMemoryPool::GetInstance().AllocTensorMem(size);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
