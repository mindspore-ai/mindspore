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
#include "runtime/device/ascend/ascend_memory_manager.h"
#include "runtime/device/ascend/ascend_memory_pool.h"
#include "utils/ms_context.h"
#include "runtime/mem.h"
namespace mindspore {
namespace device {
namespace ascend {
constexpr uint64_t kAscendDeviceMemGB = 30;
constexpr uint64_t kMemSizeGB = 30;
constexpr uint64_t kAscendDeviceMemSize = (kAscendDeviceMemGB << kMemSizeGB);

void AscendMemoryManager::MallocDeviceMemory() {
  auto context_mem = GetDeviceMemSizeFromContext();
  device_mem_size_ = context_mem == 0 ? kAscendDeviceMemSize : context_mem;
  auto ret = rtMalloc(reinterpret_cast<void **>(&device_mem_base_), device_mem_size_, RT_MEMORY_HBM);

  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << device_mem_size_ << "] fail, ret[" << ret << "]";
  }

  AscendMemoryPool::GetInstance().set_device_mem_size(device_mem_size_);
  AscendMemoryPool::GetInstance().set_device_mem_pool_base(device_mem_base_);
  AscendMemoryPool::GetInstance().set_device_mem_pool_offset(device_mem_size_);
  AscendMemoryPool::GetInstance().set_graph_dynamic_mem_offset(dynamic_mem_offset_);
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
  if (gb_var > kAscendDeviceMemGB || gb_var == 0) {
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
}

void AscendMemoryManager::ResetDynamicMemory() {
  total_dynamic_size_ = 0;
  dynamic_mem_offset_ = 0;
  AscendMemoryPool::GetInstance().set_graph_dynamic_mem_offset(dynamic_mem_offset_);
}

void *AscendMemoryManager::MallocMemFromMemPool(size_t size) {
  auto align_size = GetCommonAlignSize(size);
  return AscendMemoryPool::GetInstance().AllocTensorMem(align_size);
}

uint8_t *AscendMemoryManager::MallocStaticMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }

  auto device_mem_pool_offset = AscendMemoryPool::GetInstance().device_mem_pool_offset();
  MS_LOG(INFO) << "Malloc Memory: Static, total[" << device_mem_size_ << "] (dynamic[" << total_dynamic_size_
               << "] memory pool[" << device_mem_size_ - device_mem_pool_offset << "])"
               << " malloc [" << align_size << "] communication_mem: " << communication_mem;

  if (communication_mem) {
    // create protect area [kMemAlignSize -- data -- kMemAlignSize]
    uint8_t *alloc_address = reinterpret_cast<uint8_t *>(AscendMemoryPool::GetInstance().AllocTensorMem(align_size));
    return alloc_address + kMemAlignSize;
  } else {
    return reinterpret_cast<uint8_t *>(AscendMemoryPool::GetInstance().AllocTensorMem(align_size));
  }
}

uint8_t *AscendMemoryManager::MallocDynamicMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }

  auto device_mem_pool_offset = AscendMemoryPool::GetInstance().device_mem_pool_offset();
  MS_LOG(INFO) << "Malloc Memory: Dynamic, total[" << device_mem_size_ << "] (dynamic[" << total_dynamic_size_
               << "] memory pool[" << device_mem_size_ - device_mem_pool_offset << "])"
               << " malloc [" << align_size << "] communication_mem: " << communication_mem;
  auto offset = dynamic_mem_offset_;
  auto new_offset = dynamic_mem_offset_ + align_size;
  if (new_offset >= device_mem_pool_offset) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "] (dynamic[" << total_dynamic_size_
                      << "] memory pool[" << device_mem_size_ - device_mem_pool_offset << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_dynamic_size_ += align_size;
  dynamic_mem_offset_ = new_offset;
  AscendMemoryPool::GetInstance().set_graph_dynamic_mem_offset(dynamic_mem_offset_);
  if (communication_mem) {
    // create protect area [kMemAlignSize -- data -- kMemAlignSize]
    return device_mem_base_ + offset + kMemAlignSize;
  } else {
    return device_mem_base_ + offset;
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
