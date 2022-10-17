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

#include "plugin/device/gpu/hal/device/gpu_memory_manager.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"
#include "utils/ms_context.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
namespace mindspore {
namespace device {
namespace gpu {
void *GPUMemoryManager::MallocMemFromMemPool(size_t size, bool from_persistent_mem) {
  return GPUMemoryAllocator::GetInstance().AllocTensorMem(size, from_persistent_mem);
}

void GPUMemoryManager::FreeMemFromMemPool(void *device_ptr) {
  GPUMemoryAllocator::GetInstance().FreeTensorMem(device_ptr);
}

std::vector<void *> GPUMemoryManager::MallocContinuousMemFromMemPool(const std::vector<size_t> &size_list) {
  return GPUMemoryAllocator::GetInstance().AllocContinuousTensorMem(size_list);
}

size_t GPUMemoryManager::GetAvailableMemSize() {
  auto available_mem_size = GPUMemoryAllocator::GetInstance().free_mem_size() +
                            GPUMemoryAllocator::GetInstance().TotalMemStatistics() -
                            GPUMemoryAllocator::GetInstance().TotalUsedMemStatistics();
  return available_mem_size;
}

bool GPUMemoryManager::MallocContinuousMemFromMemPool(const DeviceAddressPtrList &addr_list, size_t total_size,
                                                      std::vector<size_t> size_list) {
  auto device_ptr_list = MallocContinuousMemFromMemPool(size_list);
  if (device_ptr_list.empty()) {
    return false;
  }
  if (addr_list.size() != device_ptr_list.size()) {
    MS_LOG(EXCEPTION) << "The size of device list is not equal to the size of address list.";
  }
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  bool need_sync_stream = false;
  for (size_t i = 0; i < addr_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(addr_list[i]);
    auto old_addr = addr_list[i]->ptr_;
    auto new_addr = device_ptr_list[i];
    MS_EXCEPTION_IF_NULL(new_addr);
    if (old_addr != nullptr) {
      need_sync_stream = true;
      CHECK_OP_RET_WITH_EXCEPT(
        GPUDeviceManager::GetInstance().CopyDeviceMemToDeviceAsync(new_addr, old_addr, size_list[i], stream),
        "Failed to copyHostMemToDeviceAsync.");
      FreeMemFromMemPool(old_addr);
    }
    addr_list[i]->ptr_ = new_addr;
    addr_list[i]->size_ = size_list[i];
    addr_list[i]->from_mem_pool_ = true;
  }
  if (need_sync_stream) {
    return GPUDeviceManager::GetInstance().SyncStream(stream);
  }
  return true;
}

void GPUMemoryManager::Initialize() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_addr = MallocMemFromMemPool(1, false);
  if (!device_addr) {
    MS_LOG(EXCEPTION) << "Dynamic memory pool init error.";
  }
  FreeMemFromMemPool(device_addr);
}

void GPUMemoryManager::Finalize() { GPUMemoryAllocator::GetInstance().ReleaseDeviceRes(); }

uint8_t *GPUMemoryManager::MallocStaticMem(size_t size, bool, uint32_t) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_ptr = MallocMemFromMemPool(size, false);
  if (device_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << size;
  }
  return AddressOffset(device_ptr, 0);
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
