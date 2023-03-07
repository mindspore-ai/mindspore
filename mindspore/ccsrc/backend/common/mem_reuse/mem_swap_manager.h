/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_SWAP_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_SWAP_MANAGER_H_

#include <map>
#include <memory>
#include <vector>
#include <utility>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "backend/common/mem_reuse/mem_copy_manager.h"

using PerformPair = std::pair<float, float>;
namespace mindspore {
namespace device {
namespace memswap {
class BACKEND_EXPORT MemSwapManager {
 public:
  explicit MemSwapManager(const MemCopyManagerPtr &mem_copy_manager)
      : tensor_size_threshold_(0),
        tensor_size_threshold_idx_(0),
        tensor_size_num_(1),
        distance_threshold_(1),
        distance_decay_step_(1),
        retreat_count_(0) {
    mem_copy_manager_ = mem_copy_manager;
  }

  MemSwapManager(const MemSwapManager &) = delete;

  MemSwapManager &operator=(const MemSwapManager &) = delete;

  ~MemSwapManager() = default;

  bool Init(const mindspore::session::KernelGraph *kernel_graph);

  void AddMemSwapTask(SwapKind swap_kind, const DeviceAddressPtr &device_address, const HostAddress &host_address,
                      bool mock, bool profiling = false, float *cost_time = nullptr) const;

  bool SyncMemCopyStream(SwapKind swap_kind) const;

  DeviceAddressPtr UpdateSwapQueue(SwapKind swap_kind, bool mock) const;

  bool RetreatSwapInfo();

  void AdjustSwapInPos(const AnfNodePtr &kernel, size_t index);

  bool trigger_swap() const { return trigger_swap_; }

  bool mem_swap_init() const { return mem_swap_initialized_; }

  void AddKernelExecutionPerform(const AnfNodePtr &kernel, float perform) const;

  float QueryKernelExecutionPerform(const AnfNodePtr &kernel) const;

  void AddKernelSwapPerform(const AnfNodePtr &kernel, size_t output_idx, const PerformPair &perform);

  const PerformPair &QueryKernelSwapPerform(const AnfNodePtr &kernel, size_t output_idx) const;

  bool QueryKernelTriggerSwap(const AnfNodePtr &kernel) const;

  bool QueryKernelTriggerSwapIn(const AnfNodePtr &kernel) const;

  size_t QueryKernelTriggerSwapInTaskNum(const AnfNodePtr &kernel) const;

  const AnfNodePtr QueryKernelByTopoOrder(size_t index) const;

  size_t QueryKernelTopoOrder(const AnfNodePtr &kernel) const;

  const MemSwapInfoSet &QueryKernelMemSwapInfo(const AnfNodePtr &kernel) const;

  void AssignHostMemory();

  const HostAddress &QueryKernelHostAddr(const AnfNodePtr &kernel, size_t output_idx) const;

  void AddKernelHostAddrIsDirty(const AnfNodePtr &kernel, size_t output_idx, bool dirty) const;

  bool QueryKernelHostAddrIsDirty(const AnfNodePtr &kernel, size_t output_idx) const;

  void ResetHostAddrIsDirty();

  bool AllocHostPinnedMem(size_t size, void **addr) const;

  void ReleaseHostPinnedMem();

  void ClearSwapQueue(bool mock) const;

  void DumpSwapInfo() const;

  void DumpUserNodes() const;

 private:
  KernelExecutionInfo &SearchKernelExecutionInfo(const AnfNodePtr &kernel) const;

  void AddSwapInfo();

  void ResetSwapInfo();

  void SaveUserKernelTopoOrder();

  bool InitSwapThreshold(size_t swap_mem_size);

  void RetreatSwapThreshold();

  void CacheCurSwapInfoSet(const AnfNodePtr &kernel);

  void AddFirstTimeMovePos(const AnfNodePtr &kernel, size_t index, bool first_time);

  bool QueryFirstTimeMovePos(const AnfNodePtr &kernel, size_t index) const;

  size_t BestSwapInPerformPos(const AnfNodePtr &trigger_kernel, const MemSwapInfo &mem_swap_info) const;

  void MoveSwapInfoPos(size_t des_pos, size_t src_pos, const MemSwapInfo &mem_swap_info);

  void AddKernelMemSwapInfo(const AnfNodePtr &kernel, const MemSwapInfo &mem_swap_info);

  void RemoveKernelMemSwapInfo(const AnfNodePtr &kernel, const MemSwapInfo &mem_swap_info);

  bool CheckDistanceBetweenKernels(const TensorInfo &tensor_info) const;

  std::vector<std::pair<size_t, size_t>> CheckDistanceBetweenKernelsWithIdx(const TensorInfo &tensor_info) const;

  bool IsCommunicationRelevantOp(const AnfNodePtr &kernel) const;

  bool IsInplaceRelevantOp(const TensorInfo &tensor);

  std::vector<CNodePtr> execution_order_;
  std::vector<TensorInfo> ordered_tensors_;
  mindspore::HashMap<void *, KernelExecutionInfo> kernel_execution_info_;
  mindspore::HashMap<void *, std::map<size_t, PerformPair>> kernel_swap_perform_;
  // Key: trigger swap kernel, value: MemSwapInfoSet of kernel need to be swapped
  mindspore::HashMap<void *, MemSwapInfoSet> mem_swap_info_map_;
  std::vector<HostAddress> host_addrs_list_;

  // Key: cache kernel address, value: lists of first time move pos or not
  std::map<void *, std::vector<bool>> kernel_first_move_cache_map_;
  std::vector<MemSwapInfo> mem_swap_info_cache_list_;
  std::pair<size_t, size_t> best_and_cur_pos_cache_;

  size_t tensor_size_threshold_;
  size_t tensor_size_threshold_idx_;
  size_t tensor_size_num_;
  size_t distance_threshold_;
  size_t distance_decay_step_;
  size_t retreat_count_;

  MemCopyManagerPtr mem_copy_manager_{nullptr};
  const mindspore::session::KernelGraph *kernel_graph_{nullptr};
  bool mem_swap_initialized_{false};
  bool swap_info_already_set_{false};
  bool trigger_swap_{false};

  static constexpr size_t kDistanceInitFactor = 3;
  static constexpr size_t kDistanceLowerBound = 3;
  // The upper bound of count for searching memory swap scheme recurrently.
  static constexpr size_t kRetreatCountMax = 50;
};
using MemSwapManagerPtr = std::shared_ptr<MemSwapManager>;
}  // namespace memswap
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_SWAP_MANAGER_H_
