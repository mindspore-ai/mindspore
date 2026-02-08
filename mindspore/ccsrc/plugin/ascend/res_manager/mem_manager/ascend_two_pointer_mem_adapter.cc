/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/mem_manager/ascend_two_pointer_mem_adapter.h"
#include <algorithm>
#include <set>
#include "include/utils/utils.h"
#include "ir/func_graph.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"

#include "plugin/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
std::string AscendTwoPointerMemAdapter::DevMemDetailInfo() const {
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

bool AscendTwoPointerMemAdapter::Initialize() {
  if (initialized_) {
    return true;
  }

  (void)AscendMemAdapter::Initialize();

  if (UseSimulationApi()) {
    return true;
  }

  device_mem_base_addr_ = MallocFromRts(ms_used_hbm_size_);
  static_mem_offset_ = ms_used_hbm_size_;
  cur_dynamic_mem_offset_ = 0;
  max_dynamic_mem_offset_ = 0;
  history_max_dynamic_mem_offset_ = 0;
  initialized_ = true;
  MS_LOG(INFO) << "Ascend Memory Adapter initialize success, Memory Statistics:" << DevMemStatistics();
  return true;
}

void AscendTwoPointerMemAdapter::SimulationInitialize() {
  AscendMemAdapter::SimulationInitialize();
  static_mem_offset_ = ms_used_hbm_size_;
  cur_dynamic_mem_offset_ = 0;
  max_dynamic_mem_offset_ = 0;
  history_max_dynamic_mem_offset_ = 0;
  uint8_t simulation_addr = 0;
  device_mem_base_addr_ = &simulation_addr;
}

bool AscendTwoPointerMemAdapter::DeInitialize() {
  if (!initialized_) {
    MS_LOG(INFO) << "DeInitialize Ascend Memory Adapter when it is not initialize";
    return false;
  }

  auto ret = FreeToRts(device_mem_base_addr_, ms_used_hbm_size_);
  if (ret) {
    (void)AscendMemAdapter::DeInitialize();

    device_mem_base_addr_ = nullptr;

    cur_dynamic_mem_offset_ = 0;
    max_dynamic_mem_offset_ = 0;
    history_max_dynamic_mem_offset_ = 0;
    dynamic_memory_block_list_.clear();

    static_mem_offset_ = 0;
    static_memory_block_list_.clear();

    initialized_ = false;
  }

  return ret;
}

uint8_t *AscendTwoPointerMemAdapter::MallocStaticDevMem(size_t size, const std::string &tag) {
  std::lock_guard<std::mutex> locker(mutex_);
  if (AscendVmmAdapter::IsEnabled()) {
    MS_LOG(ERROR) << "The device virtual memory doesn't support the O2 jit level, please set "
                     "MS_ALLOC_CONF=enable_vmm:False to disable the device virtual memory.";
    return nullptr;
  }
  size = GetRoundUpAlignSize(size);
  if (!common::IsCompileSimulation() && (static_mem_offset_ < static_cast<int64_t>(size) ||
                                         (static_mem_offset_ - static_cast<int64_t>(size)) < max_dynamic_mem_offset_)) {
    MS_LOG(INFO) << DevMemDetailInfo();
    MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Out of Memory!!! Request memory size: " << size
                      << "B, Memory Statistic:" << DevMemStatistics()
                      << "\nPlease try to reduce 'batch_size' or check whether exists extra large shape. For more "
                         "details, please refer to 'Out of Memory' at https://www.mindspore.cn .";
  }
  int64_t new_static_offset = static_mem_offset_ - static_cast<int64_t>(size);
  auto memory_block_ptr = device_mem_base_addr_ + new_static_offset;
  static_mem_offset_ = new_static_offset;
  static_memory_block_list_.push_back(std::make_shared<MemoryBlock>(memory_block_ptr, size, tag));
  return memory_block_ptr;
}

bool AscendTwoPointerMemAdapter::FreeStaticDevMem(void *addr) {
  if (common::IsCompileSimulation()) {
    return true;
  }
  MS_LOG(WARNING) << "FreeStaticDevMem addr:" << addr << " is not supported in two pointer mode.";
  return false;
}

uint8_t *AscendTwoPointerMemAdapter::MallocDynamicDevMem(size_t size, const std::string &tag) {
  std::lock_guard<std::mutex> locker(mutex_);
  if (AscendVmmAdapter::IsEnabled()) {
    MS_LOG(EXCEPTION) << "VMM is enabled, can not allocate dynamic memory.";
  }
  if (!IsDisableGeKernel()) {
    MS_LOG(EXCEPTION) << "GE Kernel mod, The dynamic memory allocation is disabled.";
  }
  size = GetRoundUpAlignSize(size);
  int64_t new_dynamic_offset = cur_dynamic_mem_offset_ + static_cast<int64_t>(size);
  if (!common::IsCompileSimulation() && new_dynamic_offset > static_mem_offset_) {
    MS_LOG(INFO) << DevMemDetailInfo();
    MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Out of Memory!!! Request memory size: " << size
                      << "B, Memory Statistic:" << DevMemStatistics()
                      << "\nPlease try to reduce 'batch_size' or check whether exists extra large shape. For more "
                         "details, please refer to 'Out of Memory' at https://www.mindspore.cn .";
  }

  auto memory_block_ptr = device_mem_base_addr_ + cur_dynamic_mem_offset_;
  cur_dynamic_mem_offset_ = new_dynamic_offset;
  max_dynamic_mem_offset_ = std::max(cur_dynamic_mem_offset_, max_dynamic_mem_offset_);
  history_max_dynamic_mem_offset_ = std::max(max_dynamic_mem_offset_, history_max_dynamic_mem_offset_);
  dynamic_memory_block_list_.push_back(std::make_shared<MemoryBlock>(memory_block_ptr, size, tag));

  return memory_block_ptr;
}

void AscendTwoPointerMemAdapter::ResetDynamicMemory() {
  cur_dynamic_mem_offset_ = 0;
  if (memory::mem_pool::IsMemoryPoolRecycle()) {
    max_dynamic_mem_offset_ = 0;
  }
  if (AscendVmmAdapter::IsEnabled()) {
    AscendVmmAdapter::GetInstance().ClearAllMemory();
  } else if (AscendGmemAdapter::GetInstance().is_eager_free_enabled()) {
    AscendGmemAdapter::GetInstance().EagerFreeDeviceMem(device_mem_base_addr_, ms_used_hbm_size_);
  }
}

uint64_t AscendTwoPointerMemAdapter::FreeDevMemSize() const {
  return std::max(static_mem_offset_ - max_dynamic_mem_offset_, 0L);
}

std::string AscendTwoPointerMemAdapter::DevMemStatistics() const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::ostringstream oss;
  oss << "\nDevice MOC memory size: " << device_hbm_total_size_ / kMBToByte << "M";
  oss << "\nMindSpore Used memory size: " << ms_used_hbm_size_ / kMBToByte << "M";
  oss << "\nMindSpore memory base address: " << reinterpret_cast<void *>(device_mem_base_addr_);
  if (!context->IsKByKExecutorMode()) {
    oss << "\nTotal Static Memory size: " << (ms_used_hbm_size_ - static_mem_offset_) / kMBToByte << "M";
    oss << "\nTotal Dynamic memory size: " << history_max_dynamic_mem_offset_ / kMBToByte << "M";
  }
  if (memory::mem_pool::IsMemoryPoolRecycle()) {
    size_t max_actual = std::max(actual_peak_memory_, (ms_used_hbm_size_ - static_mem_offset_));
    oss << "\nActual peak memory usage: " << max_actual / kMBToByte << "M";
  } else if (context->IsKByKExecutorMode()) {
    oss << "\nUsed peak memory usage (without fragments): " << used_peak_memory_ / kMBToByte << "M";
    oss << "\nActual peak memory usage (with fragments): " << actual_peak_memory_ / kMBToByte << "M";
  }
  if (!context->IsKByKExecutorMode()) {
    oss << "\nDynamic memory size of this graph: " << cur_dynamic_mem_offset_ / kMBToByte << "M";
  }
  oss << std::endl;
  return oss.str();
}

size_t AscendTwoPointerMemAdapter::GetDynamicMemUpperBound(void *min_static_addr) const {
  auto max_used_hbm = GetMsUsedHbmSize();
  size_t static_offset = reinterpret_cast<uint8_t *>(min_static_addr) - device_mem_base_addr_;
  return LongToSize(max_used_hbm) - static_offset;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
