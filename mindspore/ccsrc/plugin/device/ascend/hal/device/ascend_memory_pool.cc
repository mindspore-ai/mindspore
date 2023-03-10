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
// The minimum unit size (8MB) of memory block used for dynamic extend in graph run mode.
static const size_t ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH_RUN_MODE = 8 << 20;

void AscendMemoryPool::SetMemPoolBlockSize(size_t available_device_mem_size) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  float mem_block_size = ms_context->get_param<float>(MS_CTX_MEMPOOL_BLOCK_SIZE);
  // set from context configuration
  if (!common::IsFloatEqual(mem_block_size, kDefaultMempoolBlockSize)) {
    size_t config_size = FloatToSize(mem_block_size * kGBToByte);
    if (config_size > available_device_mem_size) {
      MS_LOG(WARNING) << "Memory pool block size " << config_size
                      << " is bigger than currently available maximum memory " << available_device_mem_size
                      << ", and the actual effective value will be " << available_device_mem_size;
    }
    // Reserve 1G for persistent_mem
    if (available_device_mem_size > kDynamicMemAllocUnitSize) {
      available_device_mem_size -= kDynamicMemAllocUnitSize;
    }
    size_t real_block_size = std::min(config_size, available_device_mem_size);
    SetMemAllocUintSize(real_block_size, kDynamicMemAllocUnitSize);
    return;
  }

  // set by default configuration
  const auto graph_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode);
  const bool is_graph_run_mode = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  if (graph_mode && is_graph_run_mode) {
    SetMemAllocUintSize(ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH_RUN_MODE,
                        ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH_RUN_MODE);
  } else {
    SetMemAllocUintSize(kDynamicMemAllocUnitSize, kDynamicMemAllocUnitSize);
  }
}

size_t AscendMemoryPool::CalMemBlockAllocSize(size_t size, bool from_persistent_mem) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size) {
    MS_LOG(INFO) << "The dynamic memory pool total size is "
                 << device::ascend::AscendMemoryPool::GetInstance().TotalMemStatistics() / kMBToByte
                 << "M, total used size is "
                 << device::ascend::AscendMemoryPool::GetInstance().TotalUsedMemStatistics() / kMBToByte
                 << "M, used peak size is "
                 << device::ascend::AscendMemoryPool::GetInstance().UsedMemPeakStatistics() / kMBToByte << "M.";
    MS_LOG(INFO) << "Memory Statistics:" << AscendMemAdapter::GetInstance().DevMemStatistics();
    return 0;
  }

  size_t alloc_mem_size;
  SetMemPoolBlockSize(device_free_mem_size);
  auto alloc_mem_unit_size = MemAllocUnitSize(from_persistent_mem);
  MS_LOG(DEBUG) << "Get unit block size " << alloc_mem_unit_size;
  alloc_mem_size = alloc_mem_unit_size;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool is_graph_run_mode = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  if (is_graph_run_mode) {
    // Growing at adding alloc unit size
    while (alloc_mem_size < size) {
      alloc_mem_size = alloc_mem_size + alloc_mem_unit_size;
    }
  } else {
    // Growing at twice of alloc unit size
    constexpr size_t kDouble = 2;
    while (alloc_mem_size < size) {
      alloc_mem_size = alloc_mem_size * kDouble;
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

void AscendMemoryPool::ResetIdleMemBuf() const {
  auto fn = [this](const MemStatusManagerPtr &mem_mng) {
    MS_EXCEPTION_IF_NULL(mem_mng);
    if (mem_mng->mem_block_list_.empty()) {
      return;
    }
    for (const auto &it : mem_mng->idle_mem_buf_map_) {
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
