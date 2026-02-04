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
#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_ASCEND_RES_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_ASCEND_RES_MANAGER_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "ir/tensor.h"
#include "runtime/hardware_abstract/event/device_event.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "plugin/ascend/res_manager/allocator/shared_memory_allocator.h"
#include "plugin/ascend/res_manager/allocator/symmetric_memory_allocator.h"

namespace mindspore {
namespace device {
namespace ascend {
using TensorPtr = tensor::TensorPtr;
struct MemUceInfo {
  int device_id = 0;
  std::vector<aclrtMemUceInfo> info;
  size_t retSize = 0;
};

class ASCEND_RES_MANAGER_EXPORT PinMemoryAllocator : public AddressAllocator {
 public:
  explicit PinMemoryAllocator(std::shared_ptr<SwapManager> swap_manager) : swap_manager_(swap_manager) {}
  virtual ~PinMemoryAllocator() = default;

  void *Alloc(size_t size, uint32_t stream_id) override;
  bool Free(void *address_ptr) override;
  bool IsPinned() override;

 private:
  std::shared_ptr<SwapManager> swap_manager_{nullptr};
};

using DeviceMemInfo = std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>;
class ASCEND_RES_MANAGER_EXPORT AscendResManager : public DeviceResManager {
 public:
  AscendResManager() = default;
  ~AscendResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  void ResetDynamicMemory() override;
  void SetDeterministic() const;
  void SetDebugKernel() const;

  void SetAclDeterministic() override;

  CollectiveCommunicationLib *collective_comm_lib() const override { return collective_comm_lib_; }
  std::shared_ptr<SwapManager> swap_manager() const override { return swap_manager_; }

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
  bool LoadCollectiveCommLib() override;
  bool IsEnableVmm() const override;

  bool BindDeviceToCurrentThread(bool force_bind) const override;
  void *GetStream() const override { return AscendStreamMng::GetInstance().default_stream(); }
  void *GetCopyDataStream() const override;

  // Relevant function to allocate and free device memory of raw ptr.
  bool AllocateMemory(DeviceAddress *const &address, uint32_t stream_id = UINT32_MAX) const override;
  void *AllocateMemory(size_t size, bool from_persistent_mem, bool need_recycle, uint32_t stream_id) override;
  void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const override;
  void FreeMemory(DeviceAddress *const &address) const override;
  bool IsNotEventUsedMemory(DeviceAddress *const &address) const override;
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
  void ResetMaxMemoryReserved() override;
  void ResetMaxMemoryAllocated() override;

  size_t EmptyCache() override;

  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) override;
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) override;

  bool CreateStream(size_t *stream_id) const override;
  bool CreateStreamWithPriority(size_t *stream_id, int32_t priority) const override;
  bool DestroyStream(size_t stream_id) const override;
  size_t QueryStreamSize() const override;
  std::vector<uint32_t> GetStreamIds() const override;
  void *GetStream(size_t stream_id) const override;
  void SetCurrentStreamId(size_t stream_id) override;
  size_t GetCurrentStreamId() const override;
  bool QueryStream(size_t stream_id) const override;
  bool SyncStream(size_t stream_id = 0) const override;
  bool SyncAllStreams(bool sync_device = true) const override;
  bool SyncNotDefaultStreams() const override;
  size_t DefaultStream() const override;
  std::pair<std::vector<size_t>, std::vector<size_t>> AllocDeviceMemoryForTensorList(
    const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) override;
  tensor::TensorPtr GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                    const std::vector<size_t> &before_padding_size,
                                                    const std::vector<size_t> &after_padding_size, size_t start,
                                                    size_t end) override;
  TensorPtr GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start, size_t end) override;
  int ResetParams(const std::vector<tensor::TensorPtr> &params) const override;

  DeviceEventPtr CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) override;
  CaptureGraphPtr CreateCaptureGraph() override;
  DeviceEventPtr CreateEventWithFlag(bool enable_timing, bool blocking, bool use_extensional_api) override;
  bool DestroyEvent(const DeviceEventPtr &event) override;
  bool DestroyAllEvents() override;

  void GetDeviceLimit(int32_t device_id, uint32_t *cube_num, uint32_t *vector_num) override;
  void SetDeviceLimit(int32_t device_id, int32_t cube_num, int32_t vector_num) override;
  void GetStreamLimit(size_t stream_id, uint32_t *cube_num, uint32_t *vector_num) override;
  void SetStreamLimit(size_t stream_id, int32_t cube_num, int32_t vector_num) override;
  void ResetStreamLimit(size_t stream_id) override;
  void UseStreamResInCurrentThread(size_t stream_id) override;
  void SetEnableStreamLimit() override;
  bool single_op_multi_stream_enable() const override;
  void set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) override;
  // Only used in graph_mode with MS_DISABLE_REF_MODE, delete it when delete MS_DISABLE_REF_MODEF
  void SetCPUMemManager();

  bool GetMemUceInfo(int32_t device_id) override;
  void UceMemRepair(int32_t device_id) override;
  void StopDevice(int32_t device_id) override;
  std::vector<std::pair<device::DeviceMemPtr, size_t>> GetMemUceAddr() override;

  // Override interface for multi stream event control.
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &input_event) override;

  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) override;

  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id) override;

  bool SyncAllEvents() override;

  bool LaunchCallback(std::function<void(void)> callback_func, size_t stream_id, bool is_block = false) const override;

  void InitializeForGe() const;

  void DestroyHccl();

  void ResetStreamAndCtx() const override;

  size_t GetCommunicationStreamID() const override;

  size_t GetCommunicationStreamIDByGroup(const std::string &group) const override;

  std::shared_ptr<AddressAllocator> pin_mem_allocator() const override { return pin_mem_allocator_; }

  std::shared_ptr<AddressAllocator> shared_mem_allocator() const override { return shared_mem_allocator_; }
  std::shared_ptr<AddressAllocator> symmetric_memory_allocator() const override { return symmetric_memory_allocator_; }

  DynamicMemPool *GetMemoryPool() override;

 protected:
  uint8_t *MallocDynamicMem(size_t size, bool communication_mem) override;

 private:
  bool SyncDeviceToHost(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                        size_t stream_id, const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) const;
  bool SyncHostToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                        size_t stream_id, const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) const;
  bool SyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                          size_t stream_id, const DeviceAddressExtPtr &src_ext,
                          const DeviceAddressExtPtr &dst_ext) const;
  bool AsyncDeviceToHost(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                         size_t stream_id, const DeviceAddressExtPtr &src_ext,
                         const DeviceAddressExtPtr &dst_ext) const;
  bool AsyncHostToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                         size_t stream_id, bool keep_src, const DeviceAddressExtPtr &src_ext,
                         const DeviceAddressExtPtr &dst_ext) const;
  bool AsyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                           size_t stream_id, const DeviceAddressExtPtr &src_ext,
                           const DeviceAddressExtPtr &dst_ext) const;
  bool CopyDeviceToHostForDiffFormat(const DeviceAddress *dst_device_address, const DeviceAddress *src_device_address,
                                     size_t stream_id, const DeviceAddressExtPtr &src_ext,
                                     const DeviceAddressExtPtr &dst_ext) const;
  bool CopyDeviceToHostForDiffType(const DeviceAddress *dst_device_address, const DeviceAddress *src_device_address,
                                   size_t stream_id, const DeviceAddressExtPtr &src_ext,
                                   const DeviceAddressExtPtr &dst_ext) const;
  bool CopyHostToDeviceForDiffFormat(const DeviceAddress *dst_device_address, const DeviceAddress *src_device_address,
                                     size_t stream_id, const DeviceAddressExtPtr &src_ext,
                                     const DeviceAddressExtPtr &dst_ext) const;
  bool CopyHostToDeviceForDiffType(const DeviceAddress *dst_device_address, const DeviceAddress *src_device_address,
                                   size_t stream_id, const DeviceAddressExtPtr &src_ext,
                                   const DeviceAddressExtPtr &dst_ext) const;
  bool SyncDeviceToDeviceWithDiffFormatType(const DeviceAddressPtr &dst_device_sync,
                                            const DeviceAddressPtr &src_device_sync, size_t stream_id,
                                            const DeviceAddressExtPtr &src_ext,
                                            const DeviceAddressExtPtr &dst_ext) const;
  bool CopyHostToDevice(const DeviceAddress *dst_device_address, const DeviceAddress *src_device_address,
                        const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext, const void *src,
                        uint64_t size, aclrtMemcpyKind kind, size_t stream_id,
                        const DeviceAddressPtr src_device_sync = nullptr) const;
  bool BaseCopy(void *dst, const void *src, uint64_t size, aclrtMemcpyKind kind, size_t stream_id,
                const DeviceAddressPtr src_device_sync = nullptr) const;

 private:
  MemUceInfo mem_uce_info_;
  std::mutex mem_uce_info_mutex_;
  bool initialized_ = false;
  // The collective communication library.
  std::function<CollectiveCommunicationLib *(void)> load_comm_lib_cb_;
  CollectiveCommunicationLib *collective_comm_lib_;

  std::shared_ptr<SwapManager> swap_manager_{nullptr};
  std::shared_ptr<PinMemoryAllocator> pin_mem_allocator_{nullptr};
  std::shared_ptr<SharedMemoryAllocator> shared_mem_allocator_{nullptr};
  std::shared_ptr<SymmetricMemoryAllocator> symmetric_memory_allocator_{nullptr};
  DeviceEventPtrList device_events_{};
  std::mutex device_events_mutex_;
  uint32_t device_id_{0};
  bool enable_memory_tracker_{false};
  std::atomic<bool> enable_res_limit_ = false;
  size_t prev_set_stream_id_ = UINT32_MAX;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_ASCEND_RES_MANAGER_H_
