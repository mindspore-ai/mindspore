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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_DEVICE_RES_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_DEVICE_RES_MANAGER_H_

#include <memory>
#include <string>
#include "external/ge/ge_allocator.h"
#include "device_address/device_address.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
class GeDeviceResManager {
 public:
  GeDeviceResManager() {}
  ~GeDeviceResManager() = default;

  void Initialize();
  void Destroy();
  bool AllocateMemory(device::DeviceAddress *const &address, uint32_t stream_id = UINT32_MAX) const;
  void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const;
  void *AllocateStaticMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const;
  void *AllocateWorkSpaceMemory(size_t size) const;
  void FreeMemory(device::DeviceAddress *const &address) const;
  void FreeMemory(void *ptr) const;
  device::DeviceAddressPtr CreateDeviceAddress() const;

  size_t GetMaxUsedMemorySize() const;
  bool BindDeviceToCurrentThread(bool force_bind) const;
  void *GetStream() const;
  bool SyncStream(size_t stream_id = 0) const;
  bool SyncStream(void *stream) const;
  void *GetCopyDataStream() const;
  bool SyncCopyStream() const;

 private:
  bool initialized_ = false;
  device::DeviceResManager *res_manager_{nullptr};
};
using GeDeviceResManagerPtr = std::shared_ptr<GeDeviceResManager>;

class GeAllocator : public ::ge::Allocator {
 public:
  explicit GeAllocator(GeDeviceResManager *res_manager) : res_manager_(res_manager) {}
  ~GeAllocator() { res_manager_ = nullptr; }
  GeAllocator(const GeAllocator &) = delete;
  GeAllocator &operator=(const GeAllocator &) = delete;
  ::ge::MemBlock *Malloc(size_t size) override;
  void Free(::ge::MemBlock *block) override;
  void ResetResManager() { res_manager_ = nullptr; }

 private:
  GeDeviceResManager *res_manager_{nullptr};
};
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_DEVICE_RES_MANAGER_H_
