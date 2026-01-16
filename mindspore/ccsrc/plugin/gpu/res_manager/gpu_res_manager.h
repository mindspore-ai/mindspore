/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSR_PLUGIN_RES_MANAGER_CPU_CPU_RES_MANAGER_H_
#define MINDSPORE_CCSR_PLUGIN_RES_MANAGER_CPU_CPU_RES_MANAGER_H_
#include <cuda_runtime_api.h>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/memory_manager/swap_manager.h"
#include "plugin/gpu/res_manager/gpu_res_manager.h"

namespace mindspore {
class DeviceEvent;
using DeviceEventPtr = std::shared_ptr<DeviceEvent>;
using DeviceEventPtrList = std::vector<DeviceEventPtr>;

namespace device {
namespace gpu {
#define SUPPORTED_CAP 5.3
#define RECOMMEND_SM 7
#define BASE 10.0
using DeviceMemInfo = std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>;
class GPUResManager : public DeviceResManager {
 public:
  GPUResManager() = default;
  ~GPUResManager() override = default;

  // Set device id and initialize device resource, such as stream, cudnn and cublas handle.
  void Initialize() override;

  // Release device memory, stream, cudnn and cublas handle, etc.
  void Destroy() override;

  bool BindDeviceToCurrentThread(bool force_bind) const override;

  std::shared_ptr<void> AllocateHostMemory(size_t size) const override;

  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                               uint32_t stream_id = kDefaultStreamIndex) const override;

  bool SyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
                const DeviceAddressExtPtr &src_ext = nullptr,
                const DeviceAddressExtPtr &dst_ext = nullptr) const override;
  bool AsyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
                 bool keep_src, const DeviceAddressExtPtr &src_ext = nullptr,
                 const DeviceAddressExtPtr &dst_ext = nullptr) const override;
  bool Copy(void *dst, const void *src, uint64_t size, CopyType kind, size_t stream_id) const override;
  bool CopyDirectly(void *dst, size_t dst_size, const void *src, size_t src_size, CopyType kind) const override;

  std::pair<std::vector<size_t>, std::vector<size_t>> AllocDeviceMemoryForTensorList(
    const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) override;
  tensor::TensorPtr GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                    const std::vector<size_t> &before_padding_size,
                                                    const std::vector<size_t> &after_padding_size, size_t start,
                                                    size_t end) override;
  tensor::TensorPtr GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                 size_t end) override;

  bool CreateStream(size_t *stream_id) const override;
  bool CreateStreamWithPriority(size_t *stream_id, int32_t priority) const override;
  size_t QueryStreamSize() const override;
  std::vector<uint32_t> GetStreamIds() const override;
  void *GetStream(size_t stream_id) const;
  size_t GetCommunicationStreamID() const override;
  bool DestroyStream(size_t stream_id) const override;
  void SetCurrentStreamId(size_t stream_id) override;
  size_t GetCurrentStreamId() const override;
  bool QueryStream(size_t stream_id) const override;
  bool SyncStream(size_t stream_id) const override;
  bool SyncAllStreams(bool sync_device = true) const override;
  bool SyncNotDefaultStreams() const override;
  size_t DefaultStream() const override;

  // Create device event for runtime.
  DeviceEventPtr CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) override;

  DeviceEventPtr CreateEventWithFlag(bool enable_timing, bool blocking, bool use_extensional_api) override;
  bool DestroyEvent(const DeviceEventPtr &event) override;
  bool DestroyAllEvents() override;

  bool LoadCollectiveCommLib() override;
  mindspore::device::CollectiveCommunicationLib *collective_comm_lib() const override { return collective_comm_lib_; }

  bool single_op_multi_stream_enable() const override;
  void set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) override;

  // Relevant function to allocate and free device memory of raw ptr.
  void *AllocateMemory(size_t size, bool from_persistent_mem, bool need_recycle, uint32_t stream_id) override;
  void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const override;
  void FreeMemory(void *ptr) const override;
  void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                       const std::vector<size_t> &keep_addr_sizes) const override;

  bool AllocateMemory(DeviceAddress *const &address, uint32_t stream_id = UINT32_MAX) const;

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
  void ResetMaxMemoryReserved() override;
  void ResetMaxMemoryAllocated() override;
  bool InitDevice();

  DynamicMemPool *GetMemoryPool() override;

 private:
  bool SyncDeviceToHost(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                        size_t stream_id) const;
  bool SyncHostToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                        size_t stream_id) const;
  bool SyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                          size_t stream_id) const;
  bool AsyncDeviceToHost(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                         size_t stream_id) const;
  bool AsyncHostToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                         size_t stream_id) const;
  bool AsyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                           size_t stream_id) const;
  bool Copy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
            cudaMemcpyKind copy_type) const;

 private:
  std::shared_ptr<SwapManager> swap_manager_{nullptr};
  mindspore::device::CollectiveCommunicationLib *collective_comm_lib_;
  DeviceEventPtrList device_events_{};
  std::mutex device_events_mutex_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif
