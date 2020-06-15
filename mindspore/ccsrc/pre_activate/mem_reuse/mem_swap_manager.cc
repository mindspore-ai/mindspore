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

#include "pre_activate/mem_reuse/mem_swap_manager.h"
#include <algorithm>
#include "session/anf_runtime_algorithm.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace device {
namespace memswap {
void MemSwapManager::Init(const mindspore::session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  execution_order_ = kernel_graph->execution_order();
  size_t kernel_index = 0;
  for (const auto &kernel : execution_order_) {
    // parse topo order of kernel
    (void)kernel_execution_info_.emplace(kernel.get(), kernel_index++);
    // parse tensor info
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();

    for (size_t output_idx = 0; output_idx < AnfAlgo::GetOutputTensorNum(kernel); ++output_idx) {
      TensorInfo tensor_info = {output_sizes[output_idx], kernel, output_idx};
      ordered_tensors_.push_back(tensor_info);
    }
  }

  // parse topo order of user kernel
  SaveUserKernelTopoOrder(kernel_graph);

  sort(ordered_tensors_.begin(), ordered_tensors_.end(),
       [](const TensorInfo &a, const TensorInfo &b) { return a.tensor_size_ > b.tensor_size_; });

  auto cur_tensor_size = ordered_tensors_.front().tensor_size_;
  for (auto &tensor_info : ordered_tensors_) {
    if (cur_tensor_size != tensor_info.tensor_size_) {
      cur_tensor_size = tensor_info.tensor_size_;
      tensor_size_num_++;
    }
  }
  tensor_size_threshold_ = ordered_tensors_.front().tensor_size_;
  tensor_size_threshold_idx_ = 0;

  distance_threshold_ = kernel_index / kDistanceInitFactor;
  mem_swap_initialized_ = true;
  MS_EXCEPTION_IF_NULL(mem_copy_manager_);
  mem_copy_manager_->Init();
}

void MemSwapManager::SaveUserKernelTopoOrder(const mindspore::session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  FuncGraphManagerPtr manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  NodeUsersMap user_map = manager->node_users();
  for (const auto &kernel : execution_order_) {
    auto iter = user_map.find(kernel);
    if (iter == user_map.end()) {
      continue;
    }
    AnfNodeIndexSet node_set = iter->second;
    auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
    for (auto &node_pair : node_set) {
      auto user_kernel = node_pair.first;
      if (!AnfAlgo::IsRealCNodeKernel(user_kernel)) {
        continue;
      }

      size_t user_kernel_topo_sort = SearchKernelExecutionInfo(user_kernel).topo_order_;
      auto kernel_with_index = AnfAlgo::GetPrevNodeOutput(user_kernel, node_pair.second - 1);
      auto &output_idx = kernel_with_index.second;
      kernel_exec_info.node_users_map_[output_idx].push_back(user_kernel_topo_sort);
    }
    for (auto &node_user_pair : kernel_exec_info.node_users_map_) {
      sort(node_user_pair.second.begin(), node_user_pair.second.end());
    }
  }
}

void MemSwapManager::AddSwapInfo() {
  for (const auto &tensor : ordered_tensors_) {
    size_t tensor_size = tensor.tensor_size_;
    if (tensor_size < tensor_size_threshold_) {
      break;
    }

    size_t output_idx = tensor.output_idx_;
    const AnfNodePtr &kernel = tensor.kernel_;
    auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
    auto &node_users_map = kernel_exec_info.node_users_map_;

    auto iter = node_users_map.find(output_idx);
    if (iter == node_users_map.end()) {
      continue;
    }
    auto &node_users = iter->second;
    bool need_swap = (node_users.size() == 1 && node_users[0] - kernel_exec_info.topo_order_ >= distance_threshold_) ||
                     (node_users.size() > 1 && node_users[1] - node_users[0] >= distance_threshold_);
    if (!need_swap) {
      continue;
    }
    AddKernelNeedSwap(kernel, true);
    HostAddress host_addr;
    host_addr.size = tensor_size;
    auto ret = AllocHostPinnedMem(tensor_size, reinterpret_cast<void **>(&host_addr.addr));
    if (!ret) {
      MS_LOG(EXCEPTION) << "Alloc host pinned memory[" << tensor_size << "] failed.";
    }
    kernel_exec_info.host_addrs_[output_idx] = host_addr;
    MemSwapInfo mem_swap_out_info = {SwapKind::kDeviceToHost, kernel, output_idx};
    if (node_users.size() > 1) {
      AddKernelMemSwapInfo(execution_order_[node_users[0]], mem_swap_out_info);
      AddKernelTriggerSwap(execution_order_[node_users[0]], true);
    } else {
      AddKernelMemSwapInfo(kernel, mem_swap_out_info);
      AddKernelTriggerSwap(kernel, true);
    }

    size_t swap_in_order = node_users.size() == 1 ? node_users[0] - 1 : node_users[1] - 1;
    if (swap_in_order <= kernel_exec_info.topo_order_) {
      MS_LOG(EXCEPTION) << "Select swap in point failed for op[" << AnfAlgo::GetCNodeName(kernel) << "]";
    }
    auto swap_in_kernel = execution_order_[swap_in_order];
    MemSwapInfo mem_swap_in_info = {SwapKind::kHostToDevice, kernel, output_idx};
    AddKernelMemSwapInfo(swap_in_kernel, mem_swap_in_info);
    AddKernelTriggerSwap(swap_in_kernel, true);

    host_addrs_list_.push_back(host_addr);
  }
}

void MemSwapManager::AddMemSwapTask(SwapKind swap_kind, const DeviceAddressPtr &device_address,
                                    const HostAddress &host_address) const {
  if (swap_kind == SwapKind::kDeviceToHost) {
    mem_copy_manager_->AddMemSwapOutTask(device_address, host_address);
  } else if (swap_kind == SwapKind::kHostToDevice) {
    mem_copy_manager_->AddMemSwapInTask(device_address, host_address);
  }
}

bool MemSwapManager::SyncMemCopyStream(SwapKind swap_kind) const {
  return mem_copy_manager_->SyncMemCopyStream(swap_kind);
}

DeviceAddressPtr MemSwapManager::UpdateSwapQueue(SwapKind swap_kind) const {
  if (swap_kind == SwapKind::kDeviceToHost) {
    return mem_copy_manager_->UpdateSwapOutQueue();
  } else {
    return mem_copy_manager_->UpdateSwapInQueue();
  }
}

// retreat to find a workable swap scheme
bool MemSwapManager::RetreatSwapInfo() {
  if (!trigger_swap_) {
    trigger_swap_ = true;
  }
  if (swap_info_already_set_) {
    ResetSwapInfo();
    if (distance_threshold_ >= kDistanceLowerBound) {
      auto distance_decay_step = execution_order_.size() / kDistanceInitFactor / tensor_size_num_;
      distance_threshold_ -= (distance_decay_step > 1 ? distance_decay_step : 1);
    }

    while (tensor_size_threshold_idx_ < ordered_tensors_.size() - 1) {
      ++tensor_size_threshold_idx_;
      if (tensor_size_threshold_idx_ > ordered_tensors_[tensor_size_threshold_idx_].tensor_size_) {
        tensor_size_threshold_ = ordered_tensors_[tensor_size_threshold_idx_].tensor_size_;
        break;
      }
    }

    if (tensor_size_threshold_idx_ == ordered_tensors_.size() - 1 && distance_threshold_ < kDistanceLowerBound) {
      MS_LOG(ERROR) << "Retreat swap info failed";
      return false;
    }
  } else {
    swap_info_already_set_ = true;
  }
  AddSwapInfo();
  return true;
}

KernelExecutionInfo &MemSwapManager::SearchKernelExecutionInfo(const AnfNodePtr &kernel) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto iter = kernel_execution_info_.find(kernel.get());
  if (iter == kernel_execution_info_.end()) {
    MS_LOG(EXCEPTION) << "Can not find execution info of op[" << AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return const_cast<KernelExecutionInfo &>(iter->second);
}

void MemSwapManager::AddKernelExecutionPerform(const AnfNodePtr &kernel, float perform) {
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  kernel_exec_info.execution_perform_ = perform;
}

void MemSwapManager::AddKernelTriggerSwap(const AnfNodePtr &kernel, bool trigger_swap) {
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  kernel_exec_info.trigger_swap_ = trigger_swap;
}

void MemSwapManager::AddKernelNeedSwap(const AnfNodePtr &kernel, bool need_swap) {
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  kernel_exec_info.need_swap_ = need_swap;
}

void MemSwapManager::AddKernelSwapPerform(const AnfNodePtr &kernel, size_t output_idx,
                                          const std::pair<float, float> &perform) {
  MS_EXCEPTION_IF_NULL(kernel);
  kernel_swap_perform_[kernel.get()][output_idx] = perform;
}

void MemSwapManager::AddKernelMemSwapInfo(const AnfNodePtr &kernel, const MemSwapInfo &mem_swap_info) {
  MS_EXCEPTION_IF_NULL(kernel);
  mem_swap_info_[kernel.get()].push_back(mem_swap_info);
}

float MemSwapManager::QueryKernelExecutionPerform(const AnfNodePtr &kernel) const {
  const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  return kernel_exec_info.execution_perform_;
}

bool MemSwapManager::QueryKernelTriggerSwap(const AnfNodePtr &kernel) const {
  const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  return kernel_exec_info.trigger_swap_;
}

bool MemSwapManager::QueryKernelNeedSwap(const AnfNodePtr &kernel) const {
  const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  return kernel_exec_info.need_swap_;
}

const PerformPair &MemSwapManager::QueryKernelSwapPerform(const AnfNodePtr &kernel, size_t output_idx) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto iter_kernel = kernel_swap_perform_.find(kernel.get());
  if (iter_kernel == kernel_swap_perform_.end()) {
    MS_LOG(EXCEPTION) << "Can not find swap performance data of op[" << AnfAlgo::GetCNodeName(kernel) << "]";
  }

  auto &perform_map = iter_kernel->second;
  auto iter_output = perform_map.find(output_idx);
  if (iter_output == perform_map.end()) {
    MS_LOG(EXCEPTION) << "Can not find swap performance data of output[" << output_idx << "] of op["
                      << AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return iter_output->second;
}

const std::vector<MemSwapInfo> &MemSwapManager::QueryKernelMemSwapInfo(const AnfNodePtr &kernel) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto iter = mem_swap_info_.find(kernel.get());
  if (iter == mem_swap_info_.end()) {
    MS_LOG(EXCEPTION) << "Can not find memory swap information data of op[" << AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return iter->second;
}

void MemSwapManager::InsertSwapInBlackList(const void *device_ptr) { swap_in_blacklist_.insert(device_ptr); }

bool MemSwapManager::FindInSwapInBlackList(const void *device_ptr) const {
  auto iter = swap_in_blacklist_.find(device_ptr);
  return iter != swap_in_blacklist_.end();
}

const HostAddress &MemSwapManager::kernel_host_addr(const AnfNodePtr &kernel, size_t output_idx) const {
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  auto &host_addrs = kernel_exec_info.host_addrs_;
  auto iter = host_addrs.find(output_idx);
  if (iter == host_addrs.end()) {
    MS_LOG(EXCEPTION) << "Can not find host address of op[" << AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return iter->second;
}

bool MemSwapManager::AllocHostPinnedMem(size_t size, void **addr) const {
  return mem_copy_manager_->AllocHostPinnedMem(size, addr);
}

void MemSwapManager::ReleaseHostPinnedMem() {
  for (const auto &host_addr : host_addrs_list_) {
    if (host_addr.addr) {
      mem_copy_manager_->FreeHostPinnedMem(host_addr.addr);
    }
  }
  host_addrs_list_.clear();
}

void MemSwapManager::ClearSwapQueue() const { mem_copy_manager_->ClearSwapQueue(); }

void MemSwapManager::ResetSwapInfo() {
  ClearSwapQueue();
  for (auto &kernel_exec_info_pair : kernel_execution_info_) {
    auto &kernel_exec_info = kernel_exec_info_pair.second;
    kernel_exec_info.trigger_swap_ = false;
    kernel_exec_info.need_swap_ = false;
    kernel_exec_info.host_addrs_.clear();
  }
  ReleaseHostPinnedMem();
  swap_in_blacklist_.clear();
  mem_swap_info_.clear();
}
}  // namespace memswap
}  // namespace device
}  // namespace mindspore
