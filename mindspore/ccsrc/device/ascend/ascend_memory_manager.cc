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
#include <string>
#include "device/ascend/ascend_memory_manager.h"
#include "device/ascend/ascend_memory_pool.h"
#include "utils/context/ms_context.h"
#include "runtime/mem.h"
namespace mindspore {
namespace device {
namespace ascend {
constexpr uint64_t kAscendDeviceMemGB = 26;
constexpr uint64_t kAscendMemPoolGB = 4;
constexpr uint64_t kMemSizeGB = 30;
constexpr uint64_t kMaxMemSizeGB = 30;
constexpr uint64_t kAscendDeviceMemSize = (kAscendDeviceMemGB << kMemSizeGB);
constexpr uint64_t kAscendMemPoolSize = (kAscendMemPoolGB << kMemSizeGB);

void AscendMemoryManager::MallocDeviceMemory() {
  auto context_mem = GetDeviceMemSizeFromContext();
  device_mem_size_ = context_mem == 0 ? kAscendDeviceMemSize : context_mem;
  static_mem_offset_ = device_mem_size_;
  auto ret = rtMalloc(reinterpret_cast<void **>(&device_mem_base_), static_mem_offset_, RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << static_mem_offset_ << "] fail, ret[" << ret << "]";
  }

  if (context_mem == 0) {
    device_mem_pool_size_ = kAscendMemPoolSize;
    ret = rtMalloc(reinterpret_cast<void **>(&device_mem_pool_base_), device_mem_pool_size_, RT_MEMORY_HBM);
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << device_mem_pool_size_ << "] fail, ret[" << ret << "]";
    }
    AscendMemoryPool::GetInstance().set_device_mem_pool_base(device_mem_pool_base_);
    AscendMemoryPool::GetInstance().set_device_mem_pool_size(device_mem_pool_size_);
  }
}

uint64_t AscendMemoryManager::GetDeviceMemSizeFromContext() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto variable_memory_max_size = context->variable_memory_max_size();
  if (variable_memory_max_size == "0") {
    return 0;
  }
  MS_LOG(INFO) << "context variable_memory_max_size:" << variable_memory_max_size;
  auto pos = variable_memory_max_size.find('*');
  if (pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Invalid variable_memory_max_size";
  }
  auto gb_str = variable_memory_max_size.substr(0, pos);
  auto gb_var = std::stoull(gb_str);
  MS_LOG(INFO) << "variable_memory_max_size(GB):" << gb_var;
  if (gb_var > kMaxMemSizeGB || gb_var == 0) {
    MS_LOG(EXCEPTION) << "Invalid allocate memory size:" << gb_var << " which should be in (0-30]GB";
  }
  return gb_var << kMemSizeGB;
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
