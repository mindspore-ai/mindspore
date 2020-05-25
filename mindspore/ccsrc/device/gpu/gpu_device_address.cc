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

#include "device/gpu/gpu_device_address.h"

#include <vector>

#include "device/gpu/gpu_device_manager.h"
#include "utils/log_adapter.h"
#include "utils/context/ms_context.h"
#include "device/gpu/gpu_memory_allocator.h"

namespace mindspore {
namespace device {
namespace gpu {
bool GPUDeviceAddress::SyncDeviceToHost(const std::vector<int> &, size_t size, TypeId, void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (size != size_) {
    MS_LOG(WARNING) << "SyncDeviceToHost ignored, host size: " << size << ", device size " << size_;
    return true;
  }
  return GPUDeviceManager::GetInstance().CopyDeviceMemToHost(host_ptr, ptr_, size_);
}

bool GPUDeviceAddress::SyncHostToDevice(const std::vector<int> &, size_t, TypeId, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  if (!GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(ptr_, host_ptr, size_, stream)) {
    MS_LOG(ERROR) << "CopyHostMemToDeviceAsync failed";
    return false;
  }
  return GPUDeviceManager::GetInstance().SyncStream(stream);
}

GPUDeviceAddress::~GPUDeviceAddress() {
  if (ptr_ == nullptr) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (from_mem_pool_) {
    GPUMemoryAllocator::GetInstance().FreeTensorMem(ptr_);
    ptr_ = nullptr;
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
