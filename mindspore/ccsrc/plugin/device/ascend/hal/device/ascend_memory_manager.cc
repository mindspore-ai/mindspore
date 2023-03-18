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
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#include "utils/ms_context.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"

using mindspore::device::ascend::ProfilingManager;
using mindspore::profiler::ascend::MemoryProfiling;
#endif

namespace mindspore {
namespace device {
namespace ascend {
void AscendMemoryManager::Initialize() { (void)AscendMemAdapter::GetInstance().Initialize(); }

void AscendMemoryManager::Finalize() {
  AscendMemoryPool::GetInstance().ReleaseDeviceRes();
  (void)AscendMemAdapter::GetInstance().DeInitialize();
}

void AscendMemoryManager::ResetDynamicMemory() { AscendMemAdapter::GetInstance().ResetDynamicMemory(); }

void AscendMemoryManager::ClearGlobalIdleMem() { AscendMemoryPool::GetInstance().ResetIdleMemBuf(); }

uint64_t AscendMemoryManager::GetMsMaxMemSize() const { return AscendMemAdapter::GetInstance().MaxHbmSizeForMs(); }

uint64_t AscendMemoryManager::GetMsUsedHbmSize() const { return AscendMemAdapter::GetInstance().GetMsUsedHbmSize(); }

void *AscendMemoryManager::MallocMemFromMemPool(size_t size, bool from_persistent_mem) {
  auto align_size = GetCommonAlignSize(size);
  const auto device_addr = AscendMemoryPool::GetInstance().AllocTensorMem(align_size, from_persistent_mem);
  return device_addr;
}

void *AscendMemoryManager::MallocOverflowMemFromMemFromMemPool(size_t size, bool from_persistent_mem) {
  const auto device_addr = AscendMemoryPool::GetInstance().AllocOverflowTensorMem(size, from_persistent_mem);
  return device_addr;
}

void AscendMemoryManager::FreeMemFromMemPool(void *device_ptr) {
  AscendMemoryPool::GetInstance().FreeTensorMem(device_ptr);
}

uint8_t *AscendMemoryManager::MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  MS_LOG(INFO) << "Malloc Memory for Static: size[" << align_size << "] communication_mem:" << communication_mem;

#ifndef ENABLE_SECURITY
  if (MemoryProfiling::GetInstance().IsMemoryProfilingInitialized() && graph_id != kInvalidGraphId) {
    auto node = MemoryProfiling::GetInstance().GetGraphMemoryNode(graph_id);
    if (node == nullptr) {
      node = MemoryProfiling::GetInstance().AddGraphMemoryNode(graph_id);
      MS_LOG(INFO) << "Add graph memory node for static memory profiling, graph id is " << graph_id;
    }

    node->AddStaticMemorySize(SizeToUint(align_size));
  }
#endif

  uint8_t *alloc_address = reinterpret_cast<uint8_t *>(AscendMemoryPool::GetInstance().AllocTensorMem(align_size));
  if (alloc_address != nullptr) {
    // create protect area [kMemAlignSize -- data -- kMemAlignSize] for communication node memory
    return communication_mem ? alloc_address + kMemAlignSize : alloc_address;
  }
  MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Fail to alloc memory, size: " << align_size
                    << ", memory statistics:" << AscendMemAdapter::GetInstance().DevMemStatistics();
}

uint8_t *AscendMemoryManager::MallocDynamicMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  MS_LOG(INFO) << "Malloc Memory for Dynamic: size[" << align_size << "] communication_mem: " << communication_mem;

  uint8_t *alloc_address = reinterpret_cast<uint8_t *>(AscendMemAdapter::GetInstance().MallocDynamicDevMem(align_size));
  MS_EXCEPTION_IF_NULL(alloc_address);
  // create protect area [kMemAlignSize -- data -- kMemAlignSize] for communication node memory
  return communication_mem ? alloc_address + kMemAlignSize : alloc_address;
}

// communication memory: [512align_size + data + 512align_size]
// return the pointer to the start of data address.
uint8_t *AscendMemoryManager::MallocCommunicationMemFromMemPool(size_t size) {
  auto align_size = GetCommunicationAlignSize(size);
  uint8_t *base_ptr = reinterpret_cast<uint8_t *>(AscendMemoryPool::GetInstance().AllocTensorMem(align_size));
  if (base_ptr != nullptr) {
    return base_ptr + kMemAlignSize;
  }
  MS_LOG(EXCEPTION) << "#umsg#Framework Error Message:#umsg#Fail to alloc memory, size: " << align_size
                    << ", memory statistics:" << AscendMemAdapter::GetInstance().DevMemStatistics();
}

bool AscendMemoryManager::MallocContinuousMemFromMemPool(const DeviceAddressPtrList &addr_list, size_t /* total_size */,
                                                         std::vector<size_t> size_list) {
  auto device_ptr_list = MallocContinuousMemFromMemPool(size_list);
  if (device_ptr_list.empty()) {
    return false;
  }
  if (addr_list.size() != device_ptr_list.size()) {
    MS_LOG(EXCEPTION) << "The size of device list " << addr_list.size() << " is not equal to the size of address list "
                      << device_ptr_list.size();
  }
  for (size_t i = 0; i < addr_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(device_ptr_list[i]);
    MS_EXCEPTION_IF_NULL(addr_list[i]);
    addr_list[i]->ptr_ = device_ptr_list[i];
    addr_list[i]->from_mem_pool_ = true;
  }
  return true;
}

size_t AscendMemoryManager::GetAvailableMemSize() {
  auto available_mem_size = AscendMemoryPool::GetInstance().free_mem_size() +
                            AscendMemoryPool::GetInstance().TotalMemStatistics() -
                            AscendMemoryPool::GetInstance().TotalUsedMemStatistics();
  return available_mem_size;
}

void AscendMemoryManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  if (stream == nullptr) {
    auto ret_rt_memcpy = aclrtMemcpy(device_ptr, mem_size, host_ptr, mem_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "SwapIn aclrtMemcpy failed.";
    }
  } else {
    auto ret_rt_memcpy = aclrtMemcpyAsync(device_ptr, mem_size, host_ptr, mem_size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "SwapIn aclrtMemcpyAsync failed.";
    }
    if (rtStreamSynchronize(stream) != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
    }
  }
}

void AscendMemoryManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  if (stream == nullptr) {
    auto ret_rt_memcpy = aclrtMemcpy(host_ptr, mem_size, device_ptr, mem_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "SwapOut aclrtMemcpy failed.";
    }
  } else {
    auto ret_rt_memcpy = aclrtMemcpyAsync(host_ptr, mem_size, device_ptr, mem_size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "SwapOut aclrtMemcpyAsync failed.";
    }
    if (rtStreamSynchronize(stream) != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
    }
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
