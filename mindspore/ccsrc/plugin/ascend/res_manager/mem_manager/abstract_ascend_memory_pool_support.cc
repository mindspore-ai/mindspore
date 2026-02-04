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

#include "plugin/ascend/res_manager/mem_manager/abstract_ascend_memory_pool_support.h"

#include <algorithm>
#include <utility>

#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/utils/utils.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace device {
namespace ascend {
// The minimum unit size (8MB) of memory block used for dynamic extend in graph run mode.
static const size_t ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH_RUN_MODE = 8 << 20;
constexpr char kGlobalOverflowWorkspace[] = "GLOBAL_OVERFLOW_WORKSPACE";

void AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(size_t available_device_mem_size) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  float mem_block_size = runtime::RuntimeConf::GetInstance()->mem_block_increase_size();
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
  static bool disable_ge_kernel = IsDisableGeKernel();
  auto is_ge = AnfAlgo::IsBackendGe();
  if (disable_ge_kernel && is_ge) {
    SetMemAllocUintSize(ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH_RUN_MODE,
                        ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH_RUN_MODE);
  } else {
    SetMemAllocUintSize(kDynamicMemAllocUnitSize, kDynamicMemAllocUnitSize);
  }
}

namespace {
bool NoAdditionalMemory() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto is_cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  const auto is_multi_graph_sink = context->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
  const auto is_task_sink = context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  const auto disable_ge_kernel = IsDisableGeKernel();
  return (is_cell_reuse || is_multi_graph_sink) && is_task_sink && disable_ge_kernel;
}
}  // namespace

size_t AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size && common::IsCompileSimulation()) {
    device_free_mem_size = size;
  }
  if (device_free_mem_size < size) {
    MS_LOG(INFO) << "The device memory is not enough, the free memory size is " << device_free_mem_size
                 << ", but the alloc size is " << size;
    MS_LOG(INFO) << "The dynamic memory pool total size is " << TotalMemStatistics() / kMBToByte
                 << "M, total used size is " << TotalUsedMemStatistics() / kMBToByte << "M, used peak size is "
                 << UsedMemPeakStatistics() / kMBToByte << "M.";
    MS_LOG(INFO) << "Memory Statistics:" << AscendMemAdapter::GetInstance()->DevMemStatistics();
    return 0;
  }

  size_t alloc_mem_size;
  SetMemPoolBlockSize(device_free_mem_size);
  auto alloc_mem_unit_size = MemAllocUnitSize(from_persistent_mem);
  if (need_recycle) {
    alloc_mem_unit_size = kDynamicMemAllocUnitSize;
  }
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
  if (NoAdditionalMemory() && !need_recycle) {
    alloc_mem_size = std::min(alloc_mem_size, size);
  }
  return alloc_mem_size;
}

size_t AbstractAscendMemoryPoolSupport::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  MS_LOG(INFO) << "Malloc Memory for Pool, size: " << size;
  if (size == 0) {
    MS_LOG(EXCEPTION) << "Failed to alloc memory pool resource, the size is zero!";
  }
  *addr = AscendMemAdapter::GetInstance()->MallocStaticDevMem(size);
  if (*addr == nullptr) {
    MS_LOG(EXCEPTION) << "Alloc device memory pool address is nullptr, failed to alloc memory pool resource!";
  }
  return size;
}

size_t AbstractAscendMemoryPoolSupport::GetMaxUsedMemSize() const {
  void *min_used_addr = GetMinUsingMemoryAddr();
  if (min_used_addr == nullptr) {
    return 0;
  }
  return AscendMemAdapter::GetInstance()->GetDynamicMemUpperBound(min_used_addr);
}

size_t AbstractAscendMemoryPoolSupport::GetVmmUsedMemSize() const {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().GetAllocatedSize();
  }
  return 0;
}

const bool AbstractAscendMemoryPoolSupport::IsEnableEagerFree() const {
  return AscendGmemAdapter::GetInstance().is_eager_free_enabled();
}

const bool AbstractAscendMemoryPoolSupport::SyncAllStreams() { return AscendStreamMng::GetInstance().SyncAllStreams(); }

size_t AbstractAscendMemoryPoolSupport::AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().AllocDeviceMem(size, addr);
  } else if (IsEnableEagerFree()) {
    return AscendGmemAdapter::GetInstance().AllocDeviceMem(size, addr);
  } else {
    MS_LOG(EXCEPTION) << "Eager free and VMM are both disabled.";
  }
}

size_t AbstractAscendMemoryPoolSupport::FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().EagerFreeDeviceMem(addr, size);
  } else if (IsEnableEagerFree()) {
    return AscendGmemAdapter::GetInstance().EagerFreeDeviceMem(addr, size);
  } else {
    MS_LOG(EXCEPTION) << "Eager free and VMM are both disabled.";
  }
}

size_t AbstractAscendMemoryPoolSupport::EmptyCache() { return AscendVmmAdapter::GetInstance().EmptyCache(); }

size_t AbstractAscendMemoryPoolSupport::MmapDeviceMem(const size_t size, const DeviceMemPtr addr) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().MmapDeviceMem(size, addr, total_mem_size());
  } else if (IsEnableEagerFree()) {
    auto ret = AscendGmemAdapter::GetInstance().MmapMemory(size, addr);
    if (ret == nullptr) {
      MS_LOG(EXCEPTION) << "Mmap memory failed.";
    }
    return size;
  }
  MS_LOG(EXCEPTION) << "Eager free and VMM are both disabled.";
}

bool AbstractAscendMemoryPoolSupport::FreeDeviceMem(const DeviceMemPtr &addr) {
  MS_EXCEPTION_IF_NULL(addr);
  int64_t max_actual = ActualPeakStatistics();
  MS_LOG(INFO) << "Max actual used memory size is " << max_actual;
  AscendMemAdapter::GetInstance()->UpdateActualPeakMemory(max_actual);
  int64_t max_peak = UsedMemPeakStatistics();
  MS_LOG(INFO) << "Max peak used memory size is " << max_peak;
  AscendMemAdapter::GetInstance()->UpdateUsedPeakMemory(max_peak);
  // disable ge kernel use two pointer mem adapter, not support free.
  if (!IsEnableVmm() && !IsEnableEagerFree() && !IsDisableGeKernel()) {
    return AscendMemAdapter::GetInstance()->FreeStaticDevMem(addr);
  }
  return true;
}

void AbstractAscendMemoryPoolSupport::ResetIdleMemBuf() const {
  // Warning : This method is not in used currently, removed in next release.
}

size_t AbstractAscendMemoryPoolSupport::free_mem_size() { return AscendMemAdapter::GetInstance()->FreeDevMemSize(); }

uint64_t AbstractAscendMemoryPoolSupport::total_mem_size() const {
  static constexpr uint64_t kMaxHbmSize = 1LL << 40;
  if (common::IsCompileSimulation()) {
    return kMaxHbmSize;
  } else {
    return AscendMemAdapter::GetInstance()->MaxHbmSizeForMs();
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
