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
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include "include/runtime/hardware_abstract/memory_manager/swap_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUResManager : public DeviceResManager {
 public:
  CPUResManager() = default;
  ~CPUResManager() override = default;

  void Destroy() override;

  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                               uint32_t stream_id = kDefaultStreamIndex) const override;

  bool SyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
                const DeviceAddressExtPtr &src_ext = nullptr,
                const DeviceAddressExtPtr &dst_ext = nullptr) const override;
  bool AsyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
                 bool keep_src, const DeviceAddressExtPtr &src_ext = nullptr,
                 const DeviceAddressExtPtr &dst_ext = nullptr) const override;
  bool Copy(void *dst, const void *src, uint64_t size, CopyType kind, size_t stream_id) const override;
  bool CopyDirectly(void *dst, size_t size, const void *src, size_t stream_id, CopyType kind) const override;

  std::pair<std::vector<size_t>, std::vector<size_t>> AllocDeviceMemoryForTensorList(
    const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) override;
  tensor::TensorPtr GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                    const std::vector<size_t> &before_padding_size,
                                                    const std::vector<size_t> &after_padding_size, size_t start,
                                                    size_t end) override;
  tensor::TensorPtr GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                 size_t end) override;

  // Relevant function to allocate and free device memory of raw ptr.
  void *AllocateMemory(size_t size, bool from_persistent_mem, bool need_recycle, uint32_t stream_id) override;
  void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const override;
  void FreeMemory(void *ptr) const override;
  void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                       const std::vector<size_t> &keep_addr_sizes) const override;
  bool LoadCollectiveCommLib() override;
  CollectiveCommunicationLib *collective_comm_lib() const override;
  void ResetDynamicMemory() override;

  DynamicMemPool *GetMemoryPool() override;

 protected:
  uint8_t *MallocDynamicMem(size_t size, bool communication_mem) override;

 private:
  uint8_t *MemMalloc(size_t size);
  void MemFree() noexcept;

  size_t mem_size_{0};
  uint8_t *mem_ptr_{nullptr};
  std::map<void *, size_t> dynamic_mem_;
  std::map<void *, size_t> static_mem_;
  std::map<void *, size_t> cached_mem_;
  std::map<void *, std::shared_ptr<std::vector<uint8_t>>> mem_block_map_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif
