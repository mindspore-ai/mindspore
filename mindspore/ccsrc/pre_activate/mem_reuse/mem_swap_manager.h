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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_SWAP_MANAGER_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_SWAP_MANAGER_H_

#include <unordered_map>
#include <unordered_set>
#include <map>
#include <memory>
#include <vector>
#include <utility>
#include "pre_activate/mem_reuse/mem_copy_manager.h"

using PerformPair = std::pair<float, float>;
namespace mindspore {
namespace device {
namespace memswap {
class MemSwapManager {
 public:
  explicit MemSwapManager(const MemCopyManagerPtr &mem_copy_manager)
      : tensor_size_threshold_(0), tensor_size_threshold_idx_(0), tensor_size_num_(1), distance_threshold_(1) {
    mem_copy_manager_ = mem_copy_manager;
  }

  ~MemSwapManager() = default;

  void Init(const mindspore::session::KernelGraph *kernel_graph);

  void AddMemSwapTask(SwapKind swap_kind, const DeviceAddressPtr &device_address, const HostAddress &host_address);

  bool SyncMemCopyStream(SwapKind swap_kind);

  DeviceAddressPtr UpdateSwapQueue(SwapKind swap_kind);

  // retreat to find a workable swap scheme
  bool RetreatSwapInfo();

  bool trigger_swap() const { return trigger_swap_; }

  bool mem_swap_init() const { return mem_swap_initialized_; }

  KernelExecutionInfo &SearchKernelExecutionInfo(const AnfNodePtr &kernel) const;

  void AddKernelExecutionPerform(const AnfNodePtr &kernel, float perform);

  float QueryKernelExecutionPerform(const AnfNodePtr &kernel) const;

  void AddKernelSwapPerform(const AnfNodePtr &kernel, size_t output_idx, const PerformPair &perform);

  const PerformPair &QueryKernelSwapPerform(const AnfNodePtr &kernel, size_t output_idx) const;

  bool QueryKerneTriggerSwap(const AnfNodePtr &kernel) const;

  bool QueryKerneNeedSwap(const AnfNodePtr &kernel) const;

  const std::vector<MemSwapInfo> &QueryKerneMemSwapInfo(const AnfNodePtr &kernel) const;

  void InsertSwapInBlackList(const void *device_ptr);

  bool FindInSwapInBlackList(const void *device_ptr) const;

  const HostAddress &kernel_host_addr(const AnfNodePtr &kernel, size_t output_idx) const;

  bool AllocHostPinnedMem(size_t size, void **addr) const;

  void ReleaseHostPinnedMem();

  void ClearSwapQueue();

 private:
  MemSwapManager(const MemSwapManager &) = delete;

  MemSwapManager &operator=(const MemSwapManager &) = delete;

  void AddSwapInfo();

  void ResetSwapInfo();

  void AddKernelTriggerSwap(const AnfNodePtr &kernel, bool trigger_swap);

  void AddKernelNeedSwap(const AnfNodePtr &kernel, bool need_swap);

  void AddKernelMemSwapInfo(const AnfNodePtr &kernel, const MemSwapInfo &mem_swap_info);

  std::vector<CNodePtr> execution_order_;
  std::vector<TensorInfo> ordered_tensors_;
  std::unordered_map<void *, KernelExecutionInfo> kernel_execution_info_;
  std::unordered_map<void *, std::map<size_t, PerformPair>> kernel_swap_perform_;
  // trigger swap kernel key : MemSwapInfo of kernel need to be swapped
  std::unordered_map<void *, std::vector<MemSwapInfo>> mem_swap_info_;
  std::vector<HostAddress> host_addrs_list_;
  std::unordered_set<const void *> swap_in_blacklist_;

  size_t tensor_size_threshold_;
  size_t tensor_size_threshold_idx_;
  size_t tensor_size_num_;
  size_t distance_threshold_;

  MemCopyManagerPtr mem_copy_manager_;
  bool mem_swap_initialized_{false};
  bool swap_info_already_set_{false};
  bool trigger_swap_{false};

  static constexpr size_t kDistanceInitFactor = 3;
  static constexpr size_t kDistanceLowerBound = 3;
};
using MemSwapManagerPtr = std::shared_ptr<MemSwapManager>;
}  // namespace memswap
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_SWAP_MANAGER_H_
