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

#include "runtime/device/ascend/ascend_memory_adapter.h"

#include <algorithm>
#include "runtime/mem.h"
#include "utils/ms_context.h"
#include "graphengine/inc/external/runtime/rt_error_codes.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr uint64_t kMemSizeGB = 30;

bool AscendMemAdapter::Initialize() {
  if (initialized_) {
    return true;
  }
  size_t free_hbm_size = 0;
  rtError_t ret = rtMemGetInfoEx(RT_MEMORYINFO_HBM, &free_hbm_size, &total_hbm_size_);
  if (ret != RT_ERROR_NONE || total_hbm_size_ == 0) {
    MS_LOG(EXCEPTION) << "Get Device HBM memory size failed, ret = " << ret << ", total HBM size :" << total_hbm_size_;
  }

  max_hbm_size_for_ms_ = total_hbm_size_ * 15 / 16;  // reserved memory is 1/16 of total
  auto context_mem = GetDeviceMemSizeFromContext();
  device_mem_size_ = context_mem == 0 ? max_hbm_size_for_ms_ : context_mem;
  device_mem_base_ = MallocFromRts(device_mem_size_);
  static_mem_offset_ = device_mem_size_;
  cur_dynamic_mem_offset_ = 0;
  max_dynamic_mem_offset_ = 0;
  MS_LOG(INFO) << " Ascend Memory Adapter initialize success, Memory Statistics:" << DevMemStatistics();
  initialized_ = true;
  return true;
}

bool AscendMemAdapter::DeInitialize() {
  if (!initialized_) {
    MS_LOG(ERROR) << " DeInitialize Ascend Memory Adapter when it is not initialize";
    return false;
  }

  auto ret = FreeToRts(device_mem_base_);
  if (ret) {
    total_hbm_size_ = 0;
    max_hbm_size_for_ms_ = 0;
    device_mem_base_ = nullptr;
    device_mem_size_ = 0;

    cur_dynamic_mem_offset_ = 0;
    max_dynamic_mem_offset_ = 0;
    dynamic_memory_block_list_.clear();

    static_mem_offset_ = 0;
    static_memory_block_list_.clear();

    MS_LOG(INFO) << " Ascend Memory Adapter initialize success, statistics:" << DevMemStatistics();
    initialized_ = false;
  }

  return ret;
}

uint8_t *AscendMemAdapter::MallocStaticDevMem(size_t size, std::string tag) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto new_static_offset = static_mem_offset_ - size;
  if (new_static_offset < max_dynamic_mem_offset_) {
    MS_LOG(ERROR) << "Out of Memory!!! Request memory size: " << size << " Memory Statistic:" << DevMemStatistics()
                  << " failed! Please try to reduce 'batch_size' or check whether exists extra large shape. More "
                     "details can be found in MindSpore's FAQ with keyword 'Out of Memory'.";
    MS_LOG(ERROR) << DevMemDetailInfo();
    return nullptr;
  }

  auto memory_block_ptr = device_mem_base_ + new_static_offset;
  static_mem_offset_ = new_static_offset;
  static_memory_block_list_.push_back(std::make_shared<MemoryBlock>(memory_block_ptr, size, tag));

  return memory_block_ptr;
}

uint8_t *AscendMemAdapter::MallocDynamicDevMem(size_t size, std::string tag) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto new_dynamic_offset = cur_dynamic_mem_offset_ + size;
  if (new_dynamic_offset > static_mem_offset_) {
    MS_LOG(ERROR) << "Out of Memory!!! Request memory size: " << size << " Memory Statistic:" << DevMemStatistics()
                  << " failed! Please try to reduce 'batch_size' or check whether exists extra large shape. More "
                     "details can be found in MindSpore's FAQ with keyword 'Out of Memory'.";
    MS_LOG(ERROR) << DevMemDetailInfo();
    return nullptr;
  }

  auto memory_block_ptr = device_mem_base_ + cur_dynamic_mem_offset_;
  cur_dynamic_mem_offset_ = new_dynamic_offset;
  max_dynamic_mem_offset_ = std::max(cur_dynamic_mem_offset_, max_dynamic_mem_offset_);
  dynamic_memory_block_list_.push_back(std::make_shared<MemoryBlock>(memory_block_ptr, size, tag));

  return memory_block_ptr;
}

void AscendMemAdapter::ResetDynamicMemory() { cur_dynamic_mem_offset_ = 0; }

std::string AscendMemAdapter::DevMemStatistics() {
  std::ostringstream oss;
  oss << "\nHBM memory size: " << total_hbm_size_;
  oss << "\nAvailable HBM memory size for MS: " << max_hbm_size_for_ms_;
  oss << "\nMS memory base size: " << device_mem_size_;
  oss << "\nMS memory base address: " << reinterpret_cast<void *>(device_mem_base_);
  oss << "\nStatic Memory size: " << device_mem_size_ - static_mem_offset_;
  oss << "\nDynamic memory size of this graph: " << cur_dynamic_mem_offset_;
  oss << "\nMAX Dynamic memory size of all graph: " << max_dynamic_mem_offset_;
  oss << "\nMS Static memory offset: " << static_mem_offset_;
  oss << std::endl;
  return oss.str();
}

std::string AscendMemAdapter::DevMemDetailInfo() {
  std::ostringstream oss;
  oss << "\nMemory Detail Info:";
  oss << "\nStatic Memory Blocks:";
  oss << "\nAddress \t Size \t tag \t";
  for (const auto &blk : static_memory_block_list_) {
    oss << "\n" << blk->mem_ptr << "\t" << blk->mem_size << "\t" << blk->mem_tag;
  }

  oss << "\nDynamic Memory Blocks:";
  oss << "\nAddress \t Size \t tag \t";
  for (const auto &blk : dynamic_memory_block_list_) {
    oss << "\n" << blk->mem_ptr << "\t" << blk->mem_size << "\t" << blk->mem_tag;
  }
  return oss.str();
}

size_t AscendMemAdapter::GetDeviceMemSizeFromContext() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto variable_memory_max_size = context->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
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

  auto max_hbm_size_for_ms_GB = max_hbm_size_for_ms_ >> kMemSizeGB;
  if (gb_var > max_hbm_size_for_ms_GB || gb_var == 0) {
    MS_LOG(EXCEPTION) << "The Total Device Memory Size is " << (total_hbm_size_ >> kMemSizeGB)
                      << " GB, variable_memory_max_size should be in range (0-" << max_hbm_size_for_ms_GB
                      << "]GB, but got " << gb_var
                      << "GB, please set the context key 'variable_memory_max_size' in valid range.";
  }
  return gb_var << kMemSizeGB;
}

uint8_t *AscendMemAdapter::MallocFromRts(size_t size) {
  uint8_t *ptr = nullptr;
  auto ret = rtMalloc(reinterpret_cast<void **>(&ptr), size, RT_MEMORY_HBM);
  if (ret != ACL_RT_SUCCESS) {
    if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      unsigned int device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      size_t free = 0;
      size_t total = 0;
      (void)rtMemGetInfoEx(RT_MEMORYINFO_HBM, &free, &total);
      MS_LOG(EXCEPTION) << "Malloc device memory failed, size[" << size << "], ret[" << ret << "], "
                        << "Device " << device_id << " Available HBM size:" << total << " free size:" << free
                        << " may be other processes occupying this card, check as: ps -ef|grep python";
    } else {
      MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << size << "] fail, ret[" << ret << "]";
    }
  } else {
    MS_LOG(INFO) << "Call rtMalloc to allocate device memory Success, size : " << size
                 << " bytes , address : " << reinterpret_cast<void *>(ptr);
  }
  return ptr;
}

bool AscendMemAdapter::FreeToRts(void *devPtr) {
  if (devPtr != nullptr) {
    auto ret = rtFree(devPtr);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "rtFree mem [" << devPtr << "] fail, ret[" << ret << "]";
      return false;
    }
  }
  return true;
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
