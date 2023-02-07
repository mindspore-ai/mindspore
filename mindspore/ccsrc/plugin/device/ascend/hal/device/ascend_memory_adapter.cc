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

#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"

#include <algorithm>
#include "ir/func_graph.h"
#include "runtime/mem.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"
#include "graphengine/inc/external/runtime/rt_error_codes.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr uint64_t kAscendMemAlignSize = 512;
constexpr double kMSMemoryRatio = 0.9375;           // 15/16
constexpr double kReservedMemoryRatio = 0.0625;     // 1/16
constexpr size_t kPerHugePageMemorySize = 2097152;  // 2mb
constexpr size_t kExtraReservedMemory = 10485760;   // 10mb
constexpr double kHalfRatio = 0.5;
// The Ascend max available device memory is 32GB.
constexpr float kAscendMaxDeviceMemory = 32;
constexpr uint64_t kOverflowAddrSize = 512;

size_t AscendMemAdapter::GetRoundDownAlignSize(size_t input_size) {
  return (input_size / kAscendMemAlignSize) * kAscendMemAlignSize;
}

size_t AscendMemAdapter::GetRoundUpAlignSize(size_t input_size) {
  return ((input_size + kAscendMemAlignSize - 1) / kAscendMemAlignSize) * kAscendMemAlignSize;
}

bool AscendMemAdapter::Initialize() {
  if (initialized_) {
    return true;
  }

  rtError_t ret = rtMemGetInfoEx(RT_MEMORYINFO_HBM, &device_hbm_free_size_, &device_hbm_total_size_);
  if (ret != RT_ERROR_NONE || device_hbm_total_size_ == 0) {
    MS_LOG(EXCEPTION) << "Internal Error: Get Device HBM memory size failed, ret = " << ret
                      << ", total HBM size :" << device_hbm_total_size_;
  }

  if (device_hbm_free_size_ < LongToSize(DoubleToLong(device_hbm_total_size_ * kHalfRatio))) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    unsigned int device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Malloc device memory failed, free memory size is less "
                         "than half of total memory size."
                      << "Device " << device_id << " Device HBM total size:" << device_hbm_total_size_
                      << " Device HBM free size:" << device_hbm_free_size_
                      << " may be other processes occupying this card, check as: ps -ef|grep python";
  }

  // get user define max backend memory
  auto user_define_ms_size = GetDeviceMemSizeFromContext();
  auto recommend_mem_size_for_others = LongToSize(DoubleToLong(device_hbm_free_size_ * kReservedMemoryRatio));
  size_t reserved_mem_size_for_others;
  if (user_define_ms_size == 0) {
    ms_used_hbm_size_ = LongToSize(DoubleToLong(device_hbm_free_size_ * kMSMemoryRatio));
    // sub the extra reserved 10mb after rounding down the 2mb
    ms_used_hbm_size_ = (ms_used_hbm_size_ / kPerHugePageMemorySize) * kPerHugePageMemorySize - kExtraReservedMemory;
    reserved_mem_size_for_others = device_hbm_free_size_ - ms_used_hbm_size_;
  } else {
    if (user_define_ms_size >= device_hbm_free_size_) {
      MS_LOG(EXCEPTION)
        << "#umsg#Framework Error Message:#umsg#The Free Device Memory Size is "
        << (SizeToFloat(device_hbm_free_size_) / kGBToByte)
        << " GB, variable_memory_max_size/max_device_memory should be in range (0-"
        << (SizeToFloat(device_hbm_free_size_) / kMBToByte) << "]MB, but got "
        << (SizeToFloat(user_define_ms_size) / kMBToByte)
        << "MB, please set the context key 'variable_memory_max_size'/'max_device_memory' in valid range.";
    }
    ms_used_hbm_size_ = user_define_ms_size;

    reserved_mem_size_for_others = device_hbm_total_size_ - ms_used_hbm_size_;
    if (reserved_mem_size_for_others < recommend_mem_size_for_others) {
      MS_LOG(WARNING) << "Reserved memory size for other components(" << reserved_mem_size_for_others
                      << ") is less than recommend size(" << recommend_mem_size_for_others
                      << "), It may lead to Out Of Memory in HCCL or other components, Please double check context key "
                         "'variable_memory_max_size'/'max_device_memory'";
    }
  }

  ms_used_hbm_size_ = GetRoundDownAlignSize(ms_used_hbm_size_);
  MS_LOG(INFO) << "Device HBM Size:" << device_hbm_total_size_ / kMBToByte
               << "M, Device free HBM Size:" << device_hbm_free_size_ / kMBToByte
               << "M, Reserved HBM size for Other Components(HCCL/rts/etc.):"
               << reserved_mem_size_for_others / kMBToByte
               << "M, Recommend Reserved HBM size for Other Components:" << recommend_mem_size_for_others / kMBToByte
               << "M, User define MindSpore HBM Size:" << user_define_ms_size / kGBToByte
               << "G, MindSpore Used HBM Size:" << ms_used_hbm_size_ / kMBToByte << "M.";

  device_mem_base_addr_ = MallocFromRts(ms_used_hbm_size_);
  static_mem_offset_ = ms_used_hbm_size_;
  cur_dynamic_mem_offset_ = 0;
  max_dynamic_mem_offset_ = 0;
  MS_LOG(INFO) << "Ascend Memory Adapter initialize success, Memory Statistics:" << DevMemStatistics();
  initialized_ = true;
  return true;
}

bool AscendMemAdapter::DeInitialize() {
  if (!initialized_) {
    MS_LOG(INFO) << "DeInitialize Ascend Memory Adapter when it is not initialize";
    return false;
  }

  auto ret = FreeToRts(device_mem_base_addr_);
  if (ret) {
    device_hbm_total_size_ = 0;
    device_hbm_free_size_ = 0;
    max_available_ms_hbm_size_ = 0;
    device_mem_base_addr_ = nullptr;
    ms_used_hbm_size_ = 0;

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

uint8_t *AscendMemAdapter::MallocStaticDevMem(size_t size, const std::string &tag) {
  std::lock_guard<std::mutex> locker(mutex_);
  size = GetRoundUpAlignSize(size);
  auto new_static_offset = static_mem_offset_ - size;
  if (new_static_offset < max_dynamic_mem_offset_) {
    MS_LOG(INFO) << DevMemDetailInfo();
    MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Out of Memory!!! Request memory size: " << size
                      << ", Memory Statistic:" << DevMemStatistics()
                      << "\nPlease try to reduce 'batch_size' or check whether exists extra large shape. For more "
                         "details, please refer to 'Out of Memory' at https://www.mindspore.cn .";
  }

  auto memory_block_ptr = device_mem_base_addr_ + new_static_offset;
  static_mem_offset_ = new_static_offset;
  static_memory_block_list_.push_back(std::make_shared<MemoryBlock>(memory_block_ptr, size, tag));
  return memory_block_ptr;
}

uint8_t *AscendMemAdapter::MallocDynamicDevMem(size_t size, const std::string &tag) {
  std::lock_guard<std::mutex> locker(mutex_);
  size = GetRoundUpAlignSize(size);
  auto new_dynamic_offset = cur_dynamic_mem_offset_ + size;
  if (new_dynamic_offset > static_mem_offset_) {
    MS_LOG(INFO) << DevMemDetailInfo();
    MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Out of Memory!!! Request memory size: " << size
                      << ", Memory Statistic:" << DevMemStatistics()
                      << "\nPlease try to reduce 'batch_size' or check whether exists extra large shape. For more "
                         "details, please refer to 'Out of Memory' at https://www.mindspore.cn .";
  }

  auto memory_block_ptr = device_mem_base_addr_ + cur_dynamic_mem_offset_;
  cur_dynamic_mem_offset_ = new_dynamic_offset;
  max_dynamic_mem_offset_ = std::max(cur_dynamic_mem_offset_, max_dynamic_mem_offset_);
  dynamic_memory_block_list_.push_back(std::make_shared<MemoryBlock>(memory_block_ptr, size, tag));

  return memory_block_ptr;
}

uint8_t *AscendMemAdapter::MallocOverflowMem(const CNodePtr &kernel) {
  std::lock_guard<std::mutex> locker(overflow_mutex_);
  auto funcGraph = kernel->func_graph();
  MS_EXCEPTION_IF_NULL(funcGraph);
  if (overflow_memory_info_map_.find(funcGraph->ToString()) != overflow_memory_info_map_.cend()) {
    return overflow_memory_info_map_.find(funcGraph->ToString())->second;
  } else {
    auto overflow_memory_ptr = MallocStaticDevMem(kOverflowAddrSize, "overflow memory ptr");
    MS_EXCEPTION_IF_NULL(overflow_memory_ptr);
    (void)overflow_memory_info_map_.emplace(funcGraph->ToString(), overflow_memory_ptr);
    return overflow_memory_ptr;
  }
}

void AscendMemAdapter::ResetDynamicMemory() { cur_dynamic_mem_offset_ = 0; }

std::string AscendMemAdapter::DevMemStatistics() const {
  std::ostringstream oss;
  oss << "\nDevice HBM memory size: " << device_hbm_total_size_ / kMBToByte << "M";
  oss << "\nMindSpore Used memory size: " << ms_used_hbm_size_ / kMBToByte << "M";
  oss << "\nMindSpore memory base address: " << reinterpret_cast<void *>(device_mem_base_addr_);
  oss << "\nTotal Static Memory size: " << (ms_used_hbm_size_ - static_mem_offset_) / kMBToByte << "M";
  oss << "\nTotal Dynamic memory size: " << max_dynamic_mem_offset_ / kMBToByte << "M";
  oss << "\nDynamic memory size of this graph: " << cur_dynamic_mem_offset_ / kMBToByte << "M";
  oss << std::endl;
  return oss.str();
}

std::string AscendMemAdapter::DevMemDetailInfo() const {
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

size_t AscendMemAdapter::GetDeviceMemSizeFromContext() const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  size_t size_from_context;
  auto max_device_memory = context->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY);
  if (max_device_memory <= kAscendMaxDeviceMemory) {
    MS_LOG(INFO) << "context max_device_memory:" << max_device_memory;
    size_from_context = FloatToSize(max_device_memory * kGBToByte);
  } else {
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
    size_from_context = gb_var * kGBToByte;
  }

  return size_from_context;
}

uint8_t *AscendMemAdapter::MallocFromRts(size_t size) const {
  uint8_t *ptr = nullptr;
  auto ret = rtMalloc(reinterpret_cast<void **>(&ptr), size, RT_MEMORY_HBM, 0);
  if (ret != ACL_RT_SUCCESS) {
    if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      unsigned int device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      size_t free = 0;
      size_t total = 0;
      (void)rtMemGetInfoEx(RT_MEMORYINFO_HBM, &free, &total);
      MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Malloc device memory failed, size[" << size << "], ret["
                        << ret << "], "
                        << "Device " << device_id << " Available HBM size:" << total << " free size:" << free
                        << " may be other processes occupying this card, check as: ps -ef|grep python";
    } else {
      MS_EXCEPTION(DeviceProcessError) << "rtMalloc mem size[" << size << "] fail, ret[" << ret << "]";
    }
  } else {
    MS_LOG(INFO) << "Call rtMalloc to allocate device memory Success, size: " << size
                 << " bytes, address start: " << reinterpret_cast<void *>(ptr)
                 << " end: " << reinterpret_cast<void *>(ptr + size);
  }
  return ptr;
}

bool AscendMemAdapter::FreeToRts(void *devPtr) const {
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
