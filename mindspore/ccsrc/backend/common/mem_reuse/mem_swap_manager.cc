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

#include "backend/common/mem_reuse/mem_swap_manager.h"
#include <algorithm>
#include <set>
#include <string>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace device {
namespace memswap {
bool MemSwapManager::Init(const mindspore::session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  execution_order_ = kernel_graph->execution_order();
  kernel_graph_ = kernel_graph;

  size_t kernel_index = 0;
  for (const auto &kernel : execution_order_) {
    // Parse topo order of kernel
    (void)kernel_execution_info_.emplace(kernel.get(), kernel_index++);
    // Parse tensor info
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel);
    for (size_t output_idx = 0; output_idx < output_num; ++output_idx) {
      TensorInfo tensor_info = {output_sizes[output_idx], kernel, output_idx};
      ordered_tensors_.push_back(tensor_info);
    }
  }

  // Parse topo order of user kernel
  SaveUserKernelTopoOrder();

  sort(ordered_tensors_.begin(), ordered_tensors_.end(),
       [](const TensorInfo &a, const TensorInfo &b) { return a.tensor_size_ > b.tensor_size_; });

  auto cur_tensor_size = ordered_tensors_.front().tensor_size_;
  for (auto &tensor_info : ordered_tensors_) {
    if (cur_tensor_size != tensor_info.tensor_size_) {
      cur_tensor_size = tensor_info.tensor_size_;
      tensor_size_num_++;
    }
  }
  if (!InitSwapThreshold(0)) {
    return false;
  }
  mem_swap_initialized_ = true;
  MS_EXCEPTION_IF_NULL(mem_copy_manager_);
  mem_copy_manager_->Init();
  return true;
}

bool MemSwapManager::InitSwapThreshold(size_t swap_mem_size) {
  distance_threshold_ = execution_order_.size() / kDistanceInitFactor;
  distance_decay_step_ = (execution_order_.size() / kDistanceInitFactor) / tensor_size_num_;
  if (distance_decay_step_ <= 1) {
    distance_decay_step_ = 1;
  }
  tensor_size_threshold_ = ordered_tensors_.front().tensor_size_;
  tensor_size_threshold_idx_ = 0;

  size_t accumulation = 0;
  while (accumulation < swap_mem_size) {
    accumulation = 0;
    for (const auto &tensor_info : ordered_tensors_) {
      size_t tensor_size = tensor_info.tensor_size_;
      if (tensor_size < tensor_size_threshold_) {
        break;
      }
      if (!CheckDistanceBetweenKernels(tensor_info)) {
        continue;
      }

      accumulation += tensor_info.tensor_size_;
      if (accumulation >= swap_mem_size) {
        return true;
      }
    }
    RetreatSwapThreshold();
    if (tensor_size_threshold_idx_ == ordered_tensors_.size() - 1 && distance_threshold_ < kDistanceLowerBound) {
      MS_LOG(ERROR) << "Init swap threshold info failed";
      return false;
    }
  }
  return true;
}

void MemSwapManager::RetreatSwapThreshold() {
  if (distance_threshold_ >= kDistanceLowerBound) {
    bool update_one_decay_step = (distance_threshold_ > distance_decay_step_) &&
                                 (distance_threshold_ - distance_decay_step_ >= kDistanceLowerBound);
    if (update_one_decay_step) {
      distance_threshold_ -= distance_decay_step_;
    } else if (distance_threshold_ >= kDistanceLowerBound) {
      static constexpr size_t kDistanceDecayStepFactor = 4;
      size_t new_distance_decay_step = (distance_threshold_ - kDistanceLowerBound) / kDistanceDecayStepFactor;
      if (new_distance_decay_step < 1) {
        new_distance_decay_step = 1;
      }
      distance_threshold_ -= new_distance_decay_step;
    }
  }

  while (tensor_size_threshold_idx_ < ordered_tensors_.size() - 1) {
    ++tensor_size_threshold_idx_;
    if (tensor_size_threshold_ > ordered_tensors_[tensor_size_threshold_idx_].tensor_size_) {
      tensor_size_threshold_ = ordered_tensors_[tensor_size_threshold_idx_].tensor_size_;
      break;
    }
  }
}

bool MemSwapManager::CheckDistanceBetweenKernels(const TensorInfo &tensor_info) const {
  const AnfNodePtr &kernel = tensor_info.kernel_;
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  auto &node_users_map = kernel_exec_info.node_users_map_;

  auto iter = node_users_map.find(tensor_info.output_idx_);
  if (iter == node_users_map.end()) {
    return false;
  }

  auto &node_users = iter->second;
  if (node_users.front() - kernel_exec_info.topo_order_ > distance_threshold_) {
    return true;
  }

  for (size_t i = 1; i < node_users.size(); ++i) {
    if (node_users[i] - node_users[i - 1] > distance_threshold_) {
      return true;
    }
  }
  return false;
}

std::vector<std::pair<size_t, size_t>> MemSwapManager::CheckDistanceBetweenKernelsWithIdx(
  const TensorInfo &tensor_info) const {
  const AnfNodePtr &kernel = tensor_info.kernel_;
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  auto &node_users_map = kernel_exec_info.node_users_map_;
  std::vector<std::pair<size_t, size_t>> need_swap_topo_pair_list;

  auto iter = node_users_map.find(tensor_info.output_idx_);
  if (iter == node_users_map.end()) {
    return need_swap_topo_pair_list;
  }
  auto &node_users = iter->second;
  if (node_users.front() - kernel_exec_info.topo_order_ > distance_threshold_) {
    need_swap_topo_pair_list.emplace_back(kernel_exec_info.topo_order_, node_users.front());
  }

  for (size_t i = 1; i < node_users.size(); ++i) {
    if (node_users[i] - node_users[i - 1] > distance_threshold_) {
      need_swap_topo_pair_list.emplace_back(node_users[i - 1], node_users[i]);
    }
  }
  return need_swap_topo_pair_list;
}

bool MemSwapManager::IsCommunicationRelevantOp(const AnfNodePtr &kernel) const {
  MS_EXCEPTION_IF_NULL(kernel);
  if (common::AnfAlgo::IsCommunicationOp(kernel)) {
    return true;
  }

  MS_EXCEPTION_IF_NULL(kernel_graph_);
  const auto &graph_manager = kernel_graph_->manager();
  MS_EXCEPTION_IF_NULL(graph_manager);
  NodeUsersMap &user_map = graph_manager->node_users();
  auto iter = user_map.find(kernel);
  bool adjacent_with_communication_op = false;
  if (iter != user_map.end()) {
    AnfNodeIndexSet node_set = iter->second;
    adjacent_with_communication_op = std::any_of(
      node_set.begin(), node_set.end(),
      [](const std::pair<AnfNodePtr, int> &node_pair) { return common::AnfAlgo::IsCommunicationOp(node_pair.first); });
  }
  return adjacent_with_communication_op;
}

bool MemSwapManager::IsInplaceRelevantOp(const TensorInfo &tensor) {
  MS_EXCEPTION_IF_NULL(tensor.kernel_);
  if (common::AnfAlgo::IsInplaceNode(tensor.kernel_, "inplace_algo") ||
      common::AnfAlgo::IsInplaceNode(tensor.kernel_, "skip")) {
    return true;
  }

  MS_EXCEPTION_IF_NULL(kernel_graph_);
  const auto &graph_manager = kernel_graph_->manager();
  MS_EXCEPTION_IF_NULL(graph_manager);
  NodeUsersMap &user_map = graph_manager->node_users();

  auto users = user_map.find(tensor.kernel_);
  for (const auto &user : users->second) {
    if (!common::AnfAlgo::IsInplaceNode(user.first, "aggregate")) {
      continue;
    }

    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(user.first, IntToSize(user.second));
    if (tensor.output_idx_ == kernel_with_index.second) {
      MS_LOG(INFO) << " [inplace optimizer] tensor: " << tensor.kernel_->DebugString()
                   << "output idx: " << tensor.output_idx_ << " used by aggregate node: " << user.first->DebugString();
      return true;
    }
  }
  return false;
}

void MemSwapManager::SaveUserKernelTopoOrder() {
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  const auto &graph_manager = kernel_graph_->manager();
  MS_EXCEPTION_IF_NULL(graph_manager);
  NodeUsersMap &user_map = graph_manager->node_users();
  for (const auto &kernel : execution_order_) {
    auto iter = user_map.find(kernel);
    if (iter == user_map.end()) {
      continue;
    }
    AnfNodeIndexSet node_set = iter->second;
    auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
    for (auto &node_pair : node_set) {
      auto user_kernel = node_pair.first;
      if (!AnfUtils::IsRealCNodeKernel(user_kernel)) {
        continue;
      }

      if (common::AnfAlgo::IsNopNode(user_kernel)) {
        continue;
      }

      size_t user_kernel_topo_sort = SearchKernelExecutionInfo(user_kernel).topo_order_;
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(user_kernel, IntToSize(node_pair.second - 1));
      auto &output_idx = kernel_with_index.second;
      if (kernel_with_index.first.get() != kernel.get()) {
        MS_LOG(EXCEPTION) << "Save user kernel topo order failed for op[" << common::AnfAlgo::GetCNodeName(kernel)
                          << "]";
      }
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

    const AnfNodePtr &kernel = tensor.kernel_;
    bool filter = IsCommunicationRelevantOp(kernel) || IsInplaceRelevantOp(tensor);
    if (filter) {
      MS_LOG(INFO) << " [inplace optimizer] ignore swap tensor: " << kernel->DebugString() << ", index"
                   << tensor.output_idx_;
      continue;
    }

    auto need_swap_topo_pair_list = CheckDistanceBetweenKernelsWithIdx(tensor);
    if (need_swap_topo_pair_list.empty()) {
      continue;
    }
    HostAddress host_addr;
    host_addr.size = tensor_size;
    host_addr.addr = nullptr;

    size_t output_idx = tensor.output_idx_;
    auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
    kernel_exec_info.host_addrs_[output_idx] = std::make_pair(host_addr, true);

    for (auto &swap_topo_pair : need_swap_topo_pair_list) {
      size_t swap_out_order = swap_topo_pair.first;
      MemSwapInfo mem_swap_out_info = {SwapKind::kDeviceToHost, kernel_exec_info.topo_order_, output_idx,
                                       swap_out_order};
      AddKernelMemSwapInfo(execution_order_[swap_out_order], mem_swap_out_info);

      size_t swap_in_order = swap_topo_pair.second - 1;
      MemSwapInfo mem_swap_in_info = {SwapKind::kHostToDevice, kernel_exec_info.topo_order_, output_idx,
                                      swap_out_order};
      if (swap_in_order <= swap_out_order) {
        MS_LOG(EXCEPTION) << "Select swap in point failed for op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
      }
      AddKernelMemSwapInfo(execution_order_[swap_in_order], mem_swap_in_info);
    }
  }
}

void MemSwapManager::AddMemSwapTask(SwapKind swap_kind, const DeviceAddressPtr &device_address,
                                    const HostAddress &host_address, bool mock, bool profiling,
                                    float *cost_time) const {
  if (!mock) {
    if (swap_kind == SwapKind::kDeviceToHost) {
      mem_copy_manager_->AddMemSwapOutTask(device_address, host_address);
    } else if (swap_kind == SwapKind::kHostToDevice) {
      mem_copy_manager_->AddMemSwapInTask(device_address, host_address, profiling, cost_time);
    }
  }

  if (swap_kind == SwapKind::kDeviceToHost) {
    mem_copy_manager_->AddMemSwapOutTaskMock(device_address);
  } else if (swap_kind == SwapKind::kHostToDevice) {
    mem_copy_manager_->AddMemSwapInTaskMock(device_address);
  }
}

bool MemSwapManager::SyncMemCopyStream(SwapKind swap_kind) const {
  return mem_copy_manager_->SyncMemCopyStream(swap_kind);
}

DeviceAddressPtr MemSwapManager::UpdateSwapQueue(SwapKind swap_kind, bool mock) const {
  if (!mock) {
    if (swap_kind == SwapKind::kDeviceToHost) {
      return mem_copy_manager_->UpdateSwapOutQueue();
    } else {
      return mem_copy_manager_->UpdateSwapInQueue();
    }
  }

  if (swap_kind == SwapKind::kDeviceToHost) {
    return mem_copy_manager_->UpdateSwapOutQueueMock();
  } else {
    return mem_copy_manager_->UpdateSwapInQueueMock();
  }
}

// Retreat to find a workable swap scheme
bool MemSwapManager::RetreatSwapInfo() {
  if (!trigger_swap_) {
    trigger_swap_ = true;
  }
  if (retreat_count_ > kRetreatCountMax) {
    MS_LOG(ERROR) << "RetreatSwapInfo exceed upper bound of count";
    return false;
  }
  retreat_count_++;

  if (swap_info_already_set_) {
    ResetSwapInfo();
    RetreatSwapThreshold();
    if (tensor_size_threshold_idx_ == ordered_tensors_.size() - 1 && distance_threshold_ < kDistanceLowerBound) {
      return false;
    }
  } else {
    swap_info_already_set_ = true;
  }
  AddSwapInfo();
  return true;
}

void MemSwapManager::AdjustSwapInPos(const AnfNodePtr &kernel, size_t index) {
  if (kernel_first_move_cache_map_.find(kernel.get()) == kernel_first_move_cache_map_.end()) {
    CacheCurSwapInfoSet(kernel);
  }

  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  size_t kernel_pos = kernel_exec_info.topo_order_;
  auto &mem_swap_info = mem_swap_info_cache_list_[index];

  if (QueryFirstTimeMovePos(kernel, index)) {
    best_and_cur_pos_cache_.first = BestSwapInPerformPos(kernel, mem_swap_info);
    best_and_cur_pos_cache_.second = best_and_cur_pos_cache_.first;
    size_t best_pos = best_and_cur_pos_cache_.first;
    if (best_pos != kernel_pos) {
      MoveSwapInfoPos(best_pos, kernel_pos, mem_swap_info);
    }
    AddFirstTimeMovePos(kernel, index, false);
    return;
  }

  auto &cur_pos = best_and_cur_pos_cache_.second;
  if (cur_pos < kernel_pos) {
    MoveSwapInfoPos(cur_pos + 1, cur_pos, mem_swap_info);
    cur_pos++;
  }
}

void MemSwapManager::CacheCurSwapInfoSet(const AnfNodePtr &kernel) {
  if (!kernel_first_move_cache_map_.empty()) {
    kernel_first_move_cache_map_.clear();
  }
  if (!mem_swap_info_cache_list_.empty()) {
    mem_swap_info_cache_list_.clear();
  }

  auto mem_swap_info_set = QueryKernelMemSwapInfo(kernel);
  size_t swap_in_task_cnt = 0;
  for (auto &mem_swap_info : mem_swap_info_set) {
    if (mem_swap_info.swap_kind_ == SwapKind::kHostToDevice) {
      mem_swap_info_cache_list_.push_back(mem_swap_info);
      kernel_first_move_cache_map_[kernel.get()].push_back(true);
      swap_in_task_cnt++;
    }
  }
  size_t swap_in_task_num = QueryKernelTriggerSwapInTaskNum(kernel);
  if (swap_in_task_cnt != swap_in_task_num) {
    MS_LOG(EXCEPTION) << "Swap_in_task_cnt :" << swap_in_task_cnt
                      << "must equal Swap_in_task_num: " << swap_in_task_num;
  }
}

void MemSwapManager::AddFirstTimeMovePos(const AnfNodePtr &kernel, size_t index, bool first_time) {
  auto iter = kernel_first_move_cache_map_.find(kernel.get());
  if (iter == kernel_first_move_cache_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find first time move pos info of op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }
  auto &first_move_list = iter->second;
  if (index >= first_move_list.size()) {
    MS_LOG(EXCEPTION) << "Index [" << index << "] out of range";
  }
  first_move_list[index] = first_time;
}

bool MemSwapManager::QueryFirstTimeMovePos(const AnfNodePtr &kernel, size_t index) const {
  auto iter = kernel_first_move_cache_map_.find(kernel.get());
  if (iter == kernel_first_move_cache_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find first time move pos info of op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }
  const auto &first_move_list = iter->second;
  if (index >= first_move_list.size()) {
    MS_LOG(EXCEPTION) << "Index [" << index << "] out of range";
  }
  return first_move_list[index];
}

size_t MemSwapManager::BestSwapInPerformPos(const AnfNodePtr &trigger_kernel, const MemSwapInfo &mem_swap_info) const {
  auto need_swap_kernel = QueryKernelByTopoOrder(mem_swap_info.topo_order_);
  const PerformPair &perform_pair = QueryKernelSwapPerform(need_swap_kernel, mem_swap_info.output_idx_);
  float swap_in_cost_time = perform_pair.second;
  size_t swap_out_pos = mem_swap_info.swap_out_pos_;
  auto &kernel_exec_info = SearchKernelExecutionInfo(trigger_kernel);
  size_t trigger_kernel_pos = kernel_exec_info.topo_order_;
  float kernel_execution_time = 0;

  size_t pos = trigger_kernel_pos;
  for (; pos > swap_out_pos + 1; pos--) {
    auto kernel = QueryKernelByTopoOrder(pos - 1);
    if (QueryKernelTriggerSwapIn(kernel)) {
      return pos;
    }
    kernel_execution_time += QueryKernelExecutionPerform(QueryKernelByTopoOrder(pos));
    if (kernel_execution_time >= swap_in_cost_time) {
      return pos - 1;
    }
  }
  return pos;
}

void MemSwapManager::MoveSwapInfoPos(size_t dest_pos, size_t src_pos, const MemSwapInfo &mem_swap_info) {
  if (dest_pos == src_pos) {
    MS_LOG(EXCEPTION) << "destination pos can not equal source pos";
  }
  auto dest_kernel = QueryKernelByTopoOrder(dest_pos);
  auto src_kernel = QueryKernelByTopoOrder(src_pos);
  AddKernelMemSwapInfo(dest_kernel, mem_swap_info);
  RemoveKernelMemSwapInfo(src_kernel, mem_swap_info);
}

KernelExecutionInfo &MemSwapManager::SearchKernelExecutionInfo(const AnfNodePtr &kernel) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto iter = kernel_execution_info_.find(kernel.get());
  if (iter == kernel_execution_info_.end()) {
    MS_LOG(EXCEPTION) << "Can not find execution info of op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return const_cast<KernelExecutionInfo &>(iter->second);
}

void MemSwapManager::AddKernelExecutionPerform(const AnfNodePtr &kernel, float perform) const {
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  kernel_exec_info.execution_perform_ = perform;
}

void MemSwapManager::AddKernelSwapPerform(const AnfNodePtr &kernel, size_t output_idx,
                                          const std::pair<float, float> &perform) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto iter = kernel_swap_perform_.find(kernel.get());
  if (iter == kernel_swap_perform_.end()) {
    kernel_swap_perform_[kernel.get()][output_idx] = perform;
  }
}

void MemSwapManager::AddKernelMemSwapInfo(const AnfNodePtr &kernel, const MemSwapInfo &mem_swap_info) {
  MS_EXCEPTION_IF_NULL(kernel);
  (void)mem_swap_info_map_[kernel.get()].insert(mem_swap_info);
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  if (mem_swap_info.swap_kind_ == SwapKind::kDeviceToHost) {
    kernel_exec_info.trigger_swap_out_ = true;
  } else if (mem_swap_info.swap_kind_ == SwapKind::kHostToDevice) {
    kernel_exec_info.swap_in_task_num_++;
    kernel_exec_info.trigger_swap_in_ = true;
  }
}

void MemSwapManager::RemoveKernelMemSwapInfo(const AnfNodePtr &kernel, const MemSwapInfo &mem_swap_info) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (mem_swap_info.swap_kind_ == SwapKind::kHostToDevice) {
    auto map_iter = mem_swap_info_map_.find(kernel.get());
    if (map_iter == mem_swap_info_map_.end()) {
      MS_LOG(EXCEPTION) << "Can not find memory swap information of op[" << common::AnfAlgo::GetCNodeName(kernel)
                        << "]";
    }
    MemSwapInfoSet &mem_swap_info_set = map_iter->second;

    auto set_iter = mem_swap_info_set.find(mem_swap_info);
    if (set_iter == mem_swap_info_set.end()) {
      MS_LOG(EXCEPTION) << "Can not find memory swap information in mem swap info set";
    }
    mem_swap_info_set.erase(set_iter);

    auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
    if (kernel_exec_info.swap_in_task_num_ > 0) {
      kernel_exec_info.swap_in_task_num_--;
    }
    if (kernel_exec_info.swap_in_task_num_ == 0) {
      kernel_exec_info.trigger_swap_in_ = false;
    }
    if (mem_swap_info_set.empty()) {
      (void)mem_swap_info_map_.erase(kernel.get());
    }
  }
}

float MemSwapManager::QueryKernelExecutionPerform(const AnfNodePtr &kernel) const {
  const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  return kernel_exec_info.execution_perform_;
}

bool MemSwapManager::QueryKernelTriggerSwap(const AnfNodePtr &kernel) const {
  const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  return kernel_exec_info.trigger_swap_out_ || kernel_exec_info.trigger_swap_in_;
}

bool MemSwapManager::QueryKernelTriggerSwapIn(const AnfNodePtr &kernel) const {
  const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  return kernel_exec_info.trigger_swap_in_;
}

size_t MemSwapManager::QueryKernelTriggerSwapInTaskNum(const AnfNodePtr &kernel) const {
  const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  return kernel_exec_info.swap_in_task_num_;
}

const AnfNodePtr MemSwapManager::QueryKernelByTopoOrder(size_t index) const {
  if (index >= execution_order_.size()) {
    MS_LOG(EXCEPTION) << "Index [" << index << "] out of range";
  }
  return execution_order_[index];
}

size_t MemSwapManager::QueryKernelTopoOrder(const AnfNodePtr &kernel) const {
  const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  return kernel_exec_info.topo_order_;
}

const PerformPair &MemSwapManager::QueryKernelSwapPerform(const AnfNodePtr &kernel, size_t output_idx) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto iter_kernel = kernel_swap_perform_.find(kernel.get());
  if (iter_kernel == kernel_swap_perform_.end()) {
    MS_LOG(EXCEPTION) << "Can not find swap performance data of op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }

  auto &perform_map = iter_kernel->second;
  auto iter_output = perform_map.find(output_idx);
  if (iter_output == perform_map.end()) {
    MS_LOG(EXCEPTION) << "Can not find swap performance data of output[" << output_idx << "] of op["
                      << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return iter_output->second;
}

const MemSwapInfoSet &MemSwapManager::QueryKernelMemSwapInfo(const AnfNodePtr &kernel) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto iter = mem_swap_info_map_.find(kernel.get());
  if (iter == mem_swap_info_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find memory swap information of op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return iter->second;
}

void MemSwapManager::AssignHostMemory() {
  for (auto &kernel_exec_info_pair : kernel_execution_info_) {
    auto &kernel_exec_info = kernel_exec_info_pair.second;
    auto &host_addrs_map = kernel_exec_info.host_addrs_;
    for (auto &host_addr_pair : host_addrs_map) {
      auto &host_addr = host_addr_pair.second.first;
      auto ret = AllocHostPinnedMem(host_addr.size, &host_addr.addr);
      if (!ret) {
        MS_LOG(EXCEPTION) << "Alloc host pinned memory[" << host_addr.size << "] failed.";
      }
      host_addrs_list_.push_back(host_addr);
    }
  }
}

const HostAddress &MemSwapManager::QueryKernelHostAddr(const AnfNodePtr &kernel, size_t output_idx) const {
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  auto &host_addrs = kernel_exec_info.host_addrs_;
  auto iter = host_addrs.find(output_idx);
  if (iter == host_addrs.end()) {
    MS_LOG(EXCEPTION) << "Can not find host address of op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return (iter->second).first;
}

void MemSwapManager::AddKernelHostAddrIsDirty(const AnfNodePtr &kernel, size_t output_idx, bool dirty) const {
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  auto &host_addrs = kernel_exec_info.host_addrs_;
  auto iter = host_addrs.find(output_idx);
  if (iter == host_addrs.end()) {
    MS_LOG(EXCEPTION) << "Can not find host memory dirty info of op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }
  (iter->second).second = dirty;
}

bool MemSwapManager::QueryKernelHostAddrIsDirty(const AnfNodePtr &kernel, size_t output_idx) const {
  auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
  auto &host_addrs = kernel_exec_info.host_addrs_;
  auto iter = host_addrs.find(output_idx);
  if (iter == host_addrs.end()) {
    MS_LOG(EXCEPTION) << "Can not find host memory dirty info of op[" << common::AnfAlgo::GetCNodeName(kernel) << "]";
  }
  return (iter->second).second;
}

void MemSwapManager::ResetHostAddrIsDirty() {
  for (auto &kernel_exec_info_pair : kernel_execution_info_) {
    auto &kernel_exec_info = kernel_exec_info_pair.second;
    auto &host_addrs = kernel_exec_info.host_addrs_;
    for (auto &host_addr : host_addrs) {
      host_addr.second.second = true;
    }
  }
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

void MemSwapManager::ClearSwapQueue(bool mock) const {
  if (!mock) {
    mem_copy_manager_->ClearSwapQueue();
  } else {
    mem_copy_manager_->ClearSwapQueueMock();
  }
}

void MemSwapManager::ResetSwapInfo() {
  ClearSwapQueue(true);
  for (auto &kernel_exec_info_pair : kernel_execution_info_) {
    auto &kernel_exec_info = kernel_exec_info_pair.second;
    kernel_exec_info.trigger_swap_out_ = false;
    kernel_exec_info.trigger_swap_in_ = false;
    kernel_exec_info.swap_in_task_num_ = 0;
    kernel_exec_info.host_addrs_.clear();
  }
  mem_swap_info_map_.clear();
}

void MemSwapManager::DumpSwapInfo() const {
  for (auto &kernel : execution_order_) {
    if (!QueryKernelTriggerSwap(kernel)) {
      continue;
    }
    auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
    MS_LOG(WARNING) << "Trigger kernel topo order[" << kernel_exec_info.topo_order_ << "] , op name["
                    << common::AnfAlgo::GetCNodeName(kernel) << "]";

    const MemSwapInfoSet &mem_swap_info_set = QueryKernelMemSwapInfo(kernel);
    for (auto &mem_swap_info : mem_swap_info_set) {
      if (mem_swap_info.swap_kind_ == SwapKind::kDeviceToHost) {
        MS_LOG(WARNING) << "    Swap Out Task: swapped kernel topo order[" << mem_swap_info.topo_order_ << "], op name["
                        << common::AnfAlgo::GetCNodeName(QueryKernelByTopoOrder(mem_swap_info.topo_order_))
                        << "], output idx[" << mem_swap_info.output_idx_ << "]";
      } else {
        MS_LOG(WARNING) << "    Swap In Task: swapped kernel topo order[" << mem_swap_info.topo_order_ << "], op name["
                        << common::AnfAlgo::GetCNodeName(QueryKernelByTopoOrder(mem_swap_info.topo_order_))
                        << "], output idx[" << mem_swap_info.output_idx_ << "]";
      }
    }
  }
}

void MemSwapManager::DumpUserNodes() const {
  for (auto &kernel : execution_order_) {
    const auto &kernel_exec_info = SearchKernelExecutionInfo(kernel);
    const auto &node_users_map = kernel_exec_info.node_users_map_;
    MS_LOG(WARNING) << "Kernel topo order[" << kernel_exec_info.topo_order_ << "], op name["
                    << common::AnfAlgo::GetCNodeName(kernel) << "]";
    if (node_users_map.empty()) {
      MS_LOG(WARNING) << "    Kernel does not own any user node";
    }

    for (auto &item : node_users_map) {
      size_t output_idx = item.first;
      auto &node_users = item.second;
      for (auto &order : node_users) {
        MS_LOG(WARNING) << "    Output index[" << output_idx << "] tensor is used by kernel["
                        << common::AnfAlgo::GetCNodeName(QueryKernelByTopoOrder(order)) << "], topo order[" << order
                        << "]";
      }
    }
  }
}
}  // namespace memswap
}  // namespace device
}  // namespace mindspore
