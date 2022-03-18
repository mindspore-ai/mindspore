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

#include <algorithm>
#include "plugin/device/ascend/hal/device/ascend_memory_pool.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#include "runtime/mem.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace device {
namespace ascend {
// The minimum unit size (8MB) of memory block used for dynamic extend in graph task sink mode.
static const size_t ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH = 8 << 20;
constexpr float kCommonMemoryRatio = 0.9667;   // 29/30
constexpr float kPersistMemoryRatio = 0.0333;  // 1/30

void AscendMemoryPool::Init() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  const bool task_sink = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto total_size = AscendMemAdapter::GetInstance().GetMsUsedHbmSize();
  if (pynative_mode) {
    SetMemPoolBlockSize(total_size);
  } else {
    if (task_sink) {
      SetMemAllocUintSize(ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH, ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH);
    } else {
      SetMemAllocUintSize(FloatToSize(total_size * kCommonMemoryRatio), FloatToSize(total_size * kPersistMemoryRatio));
    }
  }
}

size_t AscendMemoryPool::CalMemBlockAllocSize(size_t size, bool from_persistent_mem) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size) {
    MS_LOG(WARNING) << "The dynamic memory pool total size is "
                    << device::ascend::AscendMemoryPool::GetInstance().TotalMemStatistics() / kMBToByte
                    << "M, total used size is "
                    << device::ascend::AscendMemoryPool::GetInstance().TotalUsedMemStatistics() / kMBToByte
                    << "M, used peak size is "
                    << device::ascend::AscendMemoryPool::GetInstance().UsedMemPeakStatistics() / kMBToByte << "M.";
    MS_LOG(WARNING) << "Out of Memory. Request memory size: " << size << ", device free size " << device_free_mem_size
                    << ", Memory Statistic:" << AscendMemAdapter::GetInstance().DevMemStatistics()
                    << "Please try to reduce 'batch_size' or check whether exists extra large shape. More "
                       "details can be found in MindSpore's FAQ with keyword 'Out of Memory'.";
    return 0;
  }
  size_t alloc_mem_size;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  auto alloc_mem_unit_size = MemAllocUnitSize(from_persistent_mem);
  MS_LOG(DEBUG) << "Get unit block size " << alloc_mem_unit_size;
  alloc_mem_size = alloc_mem_unit_size;
  if (pynative_mode) {
    // Growing at twice of alloc unit size
    constexpr size_t kDouble = 2;
    while (alloc_mem_size < size) {
      alloc_mem_size = alloc_mem_size * kDouble;
    }
  } else {
    // Growing at adding alloc unit size
    while (alloc_mem_size < size) {
      alloc_mem_size = alloc_mem_size + alloc_mem_unit_size;
    }
  }
  alloc_mem_size = std::min(alloc_mem_size, device_free_mem_size);
  return alloc_mem_size;
}

size_t AscendMemoryPool::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  MS_LOG(INFO) << "Malloc Memory for Pool, size: " << size;
  if (size == 0) {
    MS_LOG(EXCEPTION) << "Failed to alloc memory pool resource, the size is zero!";
  }
  *addr = AscendMemAdapter::GetInstance().MallocStaticDevMem(size);
  if (*addr == nullptr) {
    MS_LOG(EXCEPTION) << "Alloc device memory pool address is nullptr, failed to alloc memory pool resource!";
  }
  return size;
}

bool AscendMemoryPool::FreeDeviceMem(const DeviceMemPtr &addr) {
  MS_EXCEPTION_IF_NULL(addr);
  return AscendMemAdapter::GetInstance().FreeStaticDevMem(addr);
}

void AscendMemoryPool::ResetIdleMemBuf() {
  auto fn = [this](const MemStatusManagerPtr &mem_mng) {
    if (mem_mng->mem_block_list_.empty()) {
      return;
    }
    for (auto &it : mem_mng->idle_mem_buf_map_) {
      MS_EXCEPTION_IF_NULL(it.second);
      (void)rtMemset(it.second->device_addr_, it.first, 0, it.first);
    }
  };
  fn(persistent_mem());
  fn(common_mem());
}

size_t AscendMemoryPool::free_mem_size() { return AscendMemAdapter::GetInstance().FreeDevMemSize(); }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
