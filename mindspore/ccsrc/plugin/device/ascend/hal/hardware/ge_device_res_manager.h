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
#include <unordered_map>
#include "runtime/hardware/device_context.h"
#include "utils/ms_context.h"
#include "include/transform/graph_ir/types.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hardware/dummy_ascend_collective_comm_lib.h"
#ifdef ENABLE_INTERNAL_KERNELS
#include "plugin/device/ascend/hal/hardware/lowlatency_collective_comm_lib.h"
#endif
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
  explicit GeHostAddress(const KernelTensorPtr &kernel_tensor) : CPUDeviceAddress(kernel_tensor) {}
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
  GeDeviceResManager() {}
  ~GeDeviceResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                               uint32_t stream_id = kDefaultStreamIndex) const override;

  DeviceAddressPtr CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const override;
  DeviceAddressPtr CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format,
                                       TypeId type_id, const std::string &device_name, uint32_t device_id,
                                       uint32_t stream_id) const override;

  static void CreateSessionAndGraphRunner();
  void MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor, const std::string &to,
              bool blocking, bool *return_self) override;

  bool LoadCollectiveCommLib() override;

  void ResetStreamAndCtx() override;
  bool BindDeviceToCurrentThread(bool force_bind) const override;
  void *GetStream() const override {
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    return runtime_instance_->compute_stream();
  }
  void *GetCopyDataStream() const {
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    return runtime_instance_->copy_data_stream();
  }

  // Relevant function to allocate and free device memory of raw ptr.
  bool AllocateMemory(DeviceAddress *const &address, uint32_t stream_id = UINT32_MAX) const override;
  void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const override;
  void FreeMemory(void *ptr) const override;
  void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                       const std::vector<size_t> &keep_addr_sizes) const override;
  void DefragMemory() override;

  size_t GetMaxUsedMemorySize() const override;

  // Relevant function to manage memory statistics
  size_t GetTotalMemStatistics() const override;
  size_t GetTotalUsedMemStatistics() const override;
  size_t GetTotalIdleMemStatistics() const override;
  size_t GetTotalEagerFreeMemStatistics() const override;
  size_t GetUsedMemPeakStatistics() const override;
  size_t GetReservedMemPeakStatistics() const override;
  std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const override;
  std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> GetCommonMemBlocksInfoStatistics()
    const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const override;
  void ResetMaxMemoryReserved() const override;
  void ResetMaxMemoryAllocated() const override;

  transform::GeAllocatorPtr GetAllocator() { return std::make_shared<GeAllocator>(this); }

  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) override;
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) override;

  bool CreateStream(size_t *stream_id) const override;
  bool CreateStreamWithPriority(size_t *stream_id, int32_t priority) const override;
  size_t QueryStreamSize() const override;
  std::vector<uint32_t> GetStreamIds() const override;
  void *GetStream(size_t stream_id) const override;
  void SetCurrentStreamId(size_t stream_id) override;
  size_t GetCurrentStreamId() const override;
  bool QueryStream(size_t stream_id) const override;
  bool SyncStream(size_t stream_id = 0) const override;
  bool SyncAllStreams() const override;
  bool SyncNotDefaultStreams() const override;
  size_t DefaultStream() const override;

  DeviceEventPtr CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait);
  DeviceEventPtr CreateEventWithFlag(bool enable_timing, bool blocking) override;

  bool single_op_multi_stream_enable() const override;
  void set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) override;
  // Only used in graph_mode with MS_DISABLE_REF_MODE, delete it when delete MS_DISABLE_REF_MODEF
  void SetCPUMemManager();

 private:
  friend class GeGraphExecutor;
  static void GeSetContextOptions(const std::shared_ptr<MsContext> &ms_context_ptr, transform::SessionOptions *options);
  static void GeSetReuseOptions(const std::string &key, size_t num, transform::SessionOptions *options);
  KernelRuntime *runtime_instance_ = nullptr;
  // Only used in graph_mode with MS_DISABLE_REF_MODE, delete it when delete MS_DISABLE_REF_MODE
  bool is_use_cpu_memory_ = false;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_RES_MANAGER_H_
