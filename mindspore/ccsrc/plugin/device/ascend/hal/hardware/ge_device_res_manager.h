/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_RES_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_RES_MANAGER_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "utils/ms_context.h"
#include "include/transform/graph_ir/types.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "runtime/device/kernel_runtime_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
class GeHostAddress : public cpu::CPUDeviceAddress {
 public:
  GeHostAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const std::string &device_name,
                uint32_t device_id)
      : CPUDeviceAddress(ptr, size, format, type_id, device_name, device_id) {}
  DeviceType GetDeviceType() const override { return DeviceType::kAscend; }
};

class GeDeviceResManager;
class GeAllocator : public ::ge::Allocator {
 public:
  explicit GeAllocator(GeDeviceResManager *res_manager) : res_manager_(res_manager) {}
  ~GeAllocator() { res_manager_ = nullptr; }
  GeAllocator(const GeAllocator &) = delete;
  GeAllocator &operator=(const GeAllocator &) = delete;
  ::ge::MemBlock *Malloc(size_t size) override;
  void Free(::ge::MemBlock *block) override;

 private:
  GeDeviceResManager *res_manager_{nullptr};
};

class GeDeviceResManager : public DeviceResManager {
 public:
  GeDeviceResManager() : mem_manager_(nullptr) {}
  ~GeDeviceResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list) const override;

  DeviceAddressPtr CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const ShapeVector &shape, const UserDataPtr &user_data = nullptr) const override;

  static void CreateSessionAndGraphRunner();

  bool LoadCollectiveCommLib() override {
    collective_comm_lib_ = &AscendCollectiveCommLib::GetInstance();
    return true;
  }

  void ResetStreamAndCtx() override;
  bool BindDeviceToCurrentThread(bool force_bind) const override;
  void *GetStream() const {
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    return runtime_instance_->compute_stream();
  }

  // Relevant function to allocate and free device memory of raw ptr.
  bool AllocateMemory(DeviceAddress *const &address) const override;
  void *AllocateMemory(size_t size) const override;
  void FreeMemory(void *ptr) const override;
  size_t GetMaxUsedMemorySize() const override;

  transform::GeAllocatorPtr GetAllocator() { return std::make_shared<GeAllocator>(this); }

  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) override;
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) override;

  bool CreateStream(size_t *stream_id) const override;
  void *GetStream(size_t stream_id) const override;
  bool SyncStream(size_t stream_id = 0) const override;
  bool SyncAllStreams() const override;

 private:
  friend class GeGraphExecutor;
  static void GeSetContextOptions(const std::shared_ptr<MsContext> &ms_context_ptr, transform::SessionOptions *options);
  static void GeSetReuseOptions(const std::string &key, size_t num, transform::SessionOptions *options);
  std::shared_ptr<MemoryManager> mem_manager_ = nullptr;
  KernelRuntime *runtime_instance_ = nullptr;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_RES_MANAGER_H_
