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

#include "plugin/ascend/res_manager/mem_manager/ascend_dynamic_mem_adapter.h"
#include <algorithm>
#include <set>
#include "ir/func_graph.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"

#include "plugin/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
uint8_t *AscendDynamicMemAdapter::MallocStaticDevMem(size_t size, const std::string &tag) {
  std::lock_guard<std::mutex> locker(mutex_);
  if (has_alloc_size + size > LongToSize(max_available_ms_hbm_size_)) {
    MS_LOG(EXCEPTION) << "No enough memory to allocate, has_alloc_size:" << has_alloc_size << ", size:" << size
                      << ", max_available_ms_moc_size:" << max_available_ms_hbm_size_;
  }
  auto addr = MallocFromRts(size);
  if (addr != nullptr) {
    has_alloc_size += size;
    (void)static_memory_blocks_.emplace(addr, std::make_shared<MemoryBlock>(addr, size, tag));
    MS_LOG(INFO) << "MallocStaticDevMem success, size:" << size << ", tag:" << tag;
  }
  return addr;
}

bool AscendDynamicMemAdapter::FreeStaticDevMem(void *addr) {
  MS_LOG(INFO) << "FreeStaticDevMem addr:" << addr << ".";
  std::lock_guard<std::mutex> locker(mutex_);
  if (addr == nullptr) {
    MS_LOG(ERROR) << "addr is nullptr.";
    return false;
  }
  auto &&iter = static_memory_blocks_.find(addr);
  if (iter == static_memory_blocks_.end()) {
    MS_LOG(ERROR) << "addr is not in static memory blocks, addr:" << addr << ".";
    return false;
  }
  auto mem_block = iter->second;
  auto ret = FreeToRts(mem_block->mem_ptr, mem_block->mem_size);
  if (!ret) {
    MS_LOG(ERROR) << "Free memory failed.";
    return false;
  }
  MS_LOG(INFO) << "Free memory success, addr:" << addr << ", size:" << mem_block->mem_size << ".";
  has_alloc_size -= mem_block->mem_size;
  static_memory_blocks_.erase(addr);
  return true;
}

bool AscendDynamicMemAdapter::Initialize() {
  if (initialized_) {
    return true;
  }
  (void)AscendMemAdapter::Initialize();
  if (UseSimulationApi()) {
    return true;
  }
  initialized_ = true;
  MS_LOG(INFO) << "Ascend Memory Adapter initialize success, Memory Statistics:" << DevMemStatistics();
  return true;
}

bool AscendDynamicMemAdapter::DeInitialize() {
  for (const auto &[addr, blk] : static_memory_blocks_) {
    if (blk->mem_ptr != nullptr) {
      auto ret = FreeToRts(blk->mem_ptr, blk->mem_size);
      if (!ret) {
        MS_LOG(ERROR) << "Free memory failed.";
        return false;
      }
      MS_LOG(INFO) << "Free memory success, addr:" << addr << ", size:" << blk->mem_size << ", tag:" << blk->mem_tag;
    }
  }
  (void)AscendMemAdapter::DeInitialize();
  has_alloc_size = 0;
  static_memory_blocks_.clear();
  initialized_ = false;
  return true;
}

uint64_t AscendDynamicMemAdapter::FreeDevMemSize() const { return max_available_ms_hbm_size_ - has_alloc_size; }

uint8_t *AscendDynamicMemAdapter::MallocDynamicDevMem(size_t size, const std::string &) {
  MS_LOG(EXCEPTION) << "MallocDynamicDevMem is disabled.";
  return nullptr;
}

void AscendDynamicMemAdapter::ResetDynamicMemory() { MS_LOG(EXCEPTION) << "ResetDynamicMemory is disabled."; }

std::string AscendDynamicMemAdapter::DevMemStatistics() const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::ostringstream oss;
  oss << "\nDevice MOC memory size: " << device_hbm_total_size_ / kMBToByte << "M";
  oss << "\nMindSpore Used memory size: " << ms_used_hbm_size_ / kMBToByte << "M";
  auto print_actual_peak_memory =
    AscendVmmAdapter::IsEnabled() ? AscendVmmAdapter::GetInstance().GetAllocatedSize() : actual_peak_memory_;
  oss << "\nUsed peak memory usage (without fragments): " << used_peak_memory_ / kMBToByte << "M";
  oss << "\nActual peak memory usage (with fragments): " << print_actual_peak_memory / kMBToByte << "M";
  oss << std::endl;
  return oss.str();
}

size_t AscendDynamicMemAdapter::GetDynamicMemUpperBound(void *min_static_addr) const {
  MS_LOG(EXCEPTION) << "GetDynamicMemUpperBound is disabled.";
  return 0;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
