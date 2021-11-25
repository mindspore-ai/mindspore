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
#include "runtime/device/ascend/ascend_memory_pool.h"
#include "runtime/device/ascend/ascend_memory_adapter.h"
#include "runtime/mem.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace device {
namespace ascend {
// The minimum unit size (256MB) of memory block used for dynamic extend.
static const size_t ASCEND_DYNAMIC_MEM_ALLOC_UNIT_SIZE = 256 << 20;
// The minimum unit size (8MB) of memory block used for dynamic extend in graph mode.
static const size_t ASCEND_DYNAMIC_MEM_ALLOC_UNIT_SIZE_FOR_GRAPH = 8 << 20;

size_t AscendMemoryPool::CalMemBlockAllocSize(size_t size) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size) {
    MS_LOG(WARNING) << "The dynamic memory pool total size is "
                    << device::ascend::AscendMemoryPool::GetInstance().total_mem_statistics() / kMBToByte
                    << "M, total used size is "
                    << device::ascend::AscendMemoryPool::GetInstance().used_mem_statistics() / kMBToByte
                    << "M, used peak size is "
                    << device::ascend::AscendMemoryPool::GetInstance().used_mem_peak_statistics() / kMBToByte << "M.";
    MS_LOG(WARNING) << "Out of Memory. Request memory size: " << size
                    << ", Memory Statistic:" << AscendMemAdapter::GetInstance().DevMemStatistics()
                    << "Please try to reduce 'batch_size' or check whether exists extra large shape. More "
                       "details can be found in MindSpore's FAQ with keyword 'Out of Memory'.";
    return 0;
  }
  auto alloc_mem_size = ASCEND_DYNAMIC_MEM_ALLOC_UNIT_SIZE;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  if (pynative_mode) {
    // Growing at twice of alloc size
    constexpr size_t kDouble = 2;
    while (alloc_mem_size < size) {
      alloc_mem_size = alloc_mem_size * kDouble;
    }
  } else {
    alloc_mem_size = ASCEND_DYNAMIC_MEM_ALLOC_UNIT_SIZE_FOR_GRAPH;
    while (alloc_mem_size < size) {
      alloc_mem_size = alloc_mem_size + ASCEND_DYNAMIC_MEM_ALLOC_UNIT_SIZE_FOR_GRAPH;
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
  auto idle_mem_buf_map = DynamicMemPoolBestFit::global_idle_mem_buf_map();
  for (auto &it : idle_mem_buf_map) {
    MS_EXCEPTION_IF_NULL(it.second);
    (void)rtMemset(it.second->device_addr_, it.first, 0, it.first);
  }
}

size_t AscendMemoryPool::free_mem_size() { return AscendMemAdapter::GetInstance().FreeDevMemSize(); }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
