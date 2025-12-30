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

#include "backend/ms_backend/runtime/actors/hyper_offload/mem_counted_cache.h"

#include "utils/ms_context.h"
#include "utils/log_adapter.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/mem_action_mgr.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/mem_use_analyzer.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace runtime {
constexpr auto kHost = "host";
constexpr auto kDevice = "device";
constexpr auto kInput = "input";
constexpr auto kOutput = "output";
constexpr auto kSplitStr = "***************";

namespace {
void PrintCache(const TensorInfoPtrSet &cache, const std::string &cache_type) {
  MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << kSplitStr << "Print " << cache_type << " begin" << kSplitStr;
  for (const auto &cc : cache) {
    MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << cc->kernel_tensor << " " << cc->status << " " << cc->next_use_idx;
  }
  MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << kSplitStr << "Print " << cache_type << " end" << kSplitStr;
}

void PrintKernelTensorList(const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors_info,
                           const std::string &tensor_type) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "Print " << tensor_type << " tensors begin" << kSplitStr;
  for (const auto &kernel_tensor_pair : kernel_tensors_info) {
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kernel_tensor_pair.first;
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "Print " << tensor_type << " tensors end" << kSplitStr;
}

device::DeviceContext *GetAscendDeviceContextPtr() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey device_key = {device::DeviceType::kAscend, device_id};
  return device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_key);
}
}  // namespace

void MemCountedCache::SetCopyStreamId(size_t copy_out_stream_id, size_t copy_in_stream_id) {
  to_host_stream_ = copy_out_stream_id;
  to_device_stream_ = copy_in_stream_id;
}

size_t MemCountedCache::GetActualHorizon(size_t cur_idx) {
  if (conditionswitch_idxs_.empty() || cur_idx == next_conditionswitch_idx_ ||
      cur_idx + horizon_ <= next_conditionswitch_idx_) {
    return horizon_;
  }
  if (cur_idx > next_conditionswitch_idx_) {
    next_conditionswitch_idx_ = conditionswitch_idxs_.front();
    conditionswitch_idxs_.pop();
    return GetActualHorizon(cur_idx);
  }
  return next_conditionswitch_idx_ - cur_idx;
}

KernelTensorPtrPairList MemCountedCache::Offload(size_t first_offloadable_idx, size_t need_size, size_t stream_id,
                                                 const device::DeviceContext *device_context) {
  KernelTensorPtrPairList ret;
  size_t has_free_size = 0;
  bool is_first_copy_out_event = true;
  std::vector<KernelTensorPtr> to_erase;
  for (auto it = device_cache_.rbegin(); it != device_cache_.rend(); ++it) {
    const auto &kernel_tensor_info = *it;
    if (kernel_tensor_info->next_use_idx < first_offloadable_idx) {
      break;
    }
    if (kernel_tensor_info->status != TensorStatus::kReady) {
      has_free_size += kernel_tensor_info->kernel_tensor->size();
      continue;
    }
    if (std::find(cur_inputs_.begin(), cur_inputs_.end(), kernel_tensor_info->kernel_tensor) != cur_inputs_.end()) {
      continue;
    }
    if (is_first_copy_out_event) {
      auto record_wait_before_action = std::make_shared<RemoteAction>(HyperOffloadEventType::kRecordWaitPairEvent,
                                                                      nullptr, stream_id, to_host_stream_);
      (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{record_wait_before_action}, device_context);
      is_first_copy_out_event = false;
    }
    MS_EXCEPTION_IF_NULL(kernel_tensor_info->kernel_tensor);
    auto action = std::make_shared<RemoteAction>(HyperOffloadEventType::kDeviceToHost,
                                                 kernel_tensor_info->kernel_tensor, to_host_stream_);
    auto offload_pairs = mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{action}, device_context);
    if (offload_pairs.size() != 1) {
      continue;
    }
    const auto &copy_kernel_tensor_pair = offload_pairs.front();
    if (copy_kernel_tensor_pair.first == nullptr || copy_kernel_tensor_pair.second == nullptr) {
      continue;
    }
    auto record_after_action = std::make_shared<RemoteAction>(
      HyperOffloadEventType::kRecordWithMemoryEvent, copy_kernel_tensor_pair.first, to_host_stream_, stream_id);
    (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{record_after_action}, device_context);
    ret.emplace_back(copy_kernel_tensor_pair);
    auto new_info = std::make_shared<TensorInfo>(copy_kernel_tensor_pair.second, kernel_tensor_info->next_use_idx,
                                                 false, TensorStatus::kCopyingToHost, copy_kernel_tensor_pair.first);
    host_cache_.insert(new_info);
    has_free_size += kernel_tensor_info->kernel_tensor->size();
    to_erase.emplace_back(copy_kernel_tensor_pair.first);
    if (has_free_size >= need_size) {
      break;
    }
  }
  if (has_free_size < need_size) {
    MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Need offload size: " << need_size << ", but only got: " << has_free_size;
  }
  for (auto &erase_tensor : to_erase) {
    UpdateTensorStatus(erase_tensor, HyperOffloadEventType::kDeviceToHost);
  }
  return ret;
}

void MemCountedCache::InsertWaitForCopyOutData(const device::DeviceContext *device_context, size_t must_released_size,
                                               bool decrease_wait_op_num) {
  size_t release_size = 0;
  // Insert wait for all wait_num counted to zero tensor
  for (const auto &kernel_tensor_info : host_cache_) {
    if (kernel_tensor_info->status != TensorStatus::kCopyingToHost) {
      continue;
    }
    if (kernel_tensor_info->copy_out_wait_op_num == 0) {
      kernel_tensor_info->status = TensorStatus::kReady;
      auto action = std::make_shared<RemoteAction>(HyperOffloadEventType::kWaitWithMemoryEvent,
                                                   kernel_tensor_info->old_kernel_tensor, to_host_stream_,
                                                   kernel_tensor_info->old_kernel_tensor->stream_id());
      (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{action}, device_context);
      release_size += kernel_tensor_info->kernel_tensor->size();
      continue;
    }
    if (decrease_wait_op_num) {
      kernel_tensor_info->copy_out_wait_op_num--;
    }
  }
  if (must_released_size > release_size) {
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Released size not enough: " << release_size
                                   << " , must release size: " << must_released_size;
  }

  // Loop again if release size less than must insert wait size
  for (auto it = host_cache_.rbegin(); release_size < must_released_size && it != host_cache_.rend(); ++it) {
    const auto &kernel_tensor_info = *it;
    if (kernel_tensor_info->status != TensorStatus::kCopyingToHost) {
      continue;
    }
    kernel_tensor_info->status = TensorStatus::kReady;
    auto action =
      std::make_shared<RemoteAction>(HyperOffloadEventType::kWaitWithMemoryEvent, kernel_tensor_info->old_kernel_tensor,
                                     to_host_stream_, kernel_tensor_info->old_kernel_tensor->stream_id());
    (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{action}, device_context);
    kernel_tensor_info->copy_out_wait_op_num = 0;
    release_size += kernel_tensor_info->kernel_tensor->size();
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Released size: " << release_size << " , must release size: " << must_released_size;
}

size_t MemCountedCache::GetIdleMemSize() {
  device::DeviceContext *device_context = GetAscendDeviceContextPtr();
  size_t total_mem = device_context->device_res_manager_->GetTotalMemStatistics();
  size_t used_mem = device_context->device_res_manager_->GetTotalUsedMemStatistics();
  if (total_mem < used_mem) {
    MS_EXCEPTION(RuntimeError) << "Memory not enough, total mem less than used mem, total mem: " << total_mem
                               << ", used mem: " << used_mem;
  }
  size_t idle_mem = total_mem - used_mem;
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "idle_mem" << kSplitStr << ": " << idle_mem;

  return idle_mem;
}

size_t MemCountedCache::GetDeviceAvailableMemSize() {
  device::DeviceContext *device_context = GetAscendDeviceContextPtr();
  size_t total_mem = device_context->device_res_manager_->GetTotalMemStatistics();
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "total_mem" << kSplitStr << ": " << total_mem;
  if (max_mem_ < total_mem) {
    MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Device memory not enough.";
    return 0;
  }

  return max_mem_ - total_mem;
}

size_t MemCountedCache::GetUsedMemSize() {
  device::DeviceContext *device_context = GetAscendDeviceContextPtr();
  return device_context->device_res_manager_->GetTotalUsedMemStatistics();
}

size_t MemCountedCache::GetNeedOffloadSize(size_t need_size) {
  size_t device_available_size = GetDeviceAvailableMemSize();
  size_t need_offload_size = 0;
  if (device_available_size >= need_size) {
    return 0;
  }
  need_offload_size = need_size - device_available_size;
  size_t idle_size = GetIdleMemSize();
  if (idle_size >= need_offload_size) {
    return 0;
  }
  need_offload_size = need_offload_size - idle_size;
  return need_offload_size;
}

void MemCountedCache::CheckInputsAvailable(size_t idx,
                                           const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors,
                                           const device::DeviceContext *device_context) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "CheckInputsAvailable begin" << kSplitStr << ", current idx:" << idx;
  PrintKernelTensorList(kernel_tensors, kInput);
  PrintCache(device_cache_, kDevice);
  PrintCache(host_cache_, kHost);
  for (const auto &kernel_tensor_info : kernel_tensors) {
    cur_inputs_.emplace_back(kernel_tensor_info.first);
    KernelTensorPtr kernel_tensor = kernel_tensor_info.first;
    size_t next_use_idx = kernel_tensor_info.second;
    bool is_last_user = next_use_idx == SIZE_MAX;
    auto it = std::find_if(device_cache_.begin(), device_cache_.end(), [&kernel_tensor](const TensorInfoPtr &info) {
      return info->kernel_tensor.get() == kernel_tensor.get();
    });
    if (it != device_cache_.end()) {
      auto &tensor_info = *it;
      if (tensor_info->status == TensorStatus::kCopyingToDevice) {
        MS_EXCEPTION_IF_NULL(tensor_info->old_kernel_tensor);
        auto action =
          std::make_shared<RemoteAction>(HyperOffloadEventType::kWaitWithMemoryEvent, tensor_info->old_kernel_tensor,
                                         to_device_stream_, kernel_tensor->stream_id());
        (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{action}, device_context);
      }
      MS_EXCEPTION_IF_CHECK_FAIL(tensor_info->status != TensorStatus::kCopyingToHost,
                                 "Invalid kernel tensor in device cache: " + tensor_info->kernel_tensor->ToString());
      TensorInfoPtr new_tensor_info = std::make_shared<TensorInfo>(kernel_tensor, is_last_user ? idx : next_use_idx,
                                                                   is_last_user, TensorStatus::kReady);
      device_cache_.erase(it);
      device_cache_.insert(new_tensor_info);
    } else {
      auto it_host = std::find_if(host_cache_.begin(), host_cache_.end(), [&kernel_tensor](const TensorInfoPtr &info) {
        return info->kernel_tensor.get() == kernel_tensor.get();
      });
      if (it_host != host_cache_.end()) {
        MS_EXCEPTION(RuntimeError)
          << "CopyToDevice Error: can't find kernel tensor in device_cache, memory not enough.";
      }

      // Insert target kernel tensor to device cache
      TensorInfoPtr new_tensor_info = std::make_shared<TensorInfo>(kernel_tensor, is_last_user ? idx : next_use_idx,
                                                                   is_last_user, TensorStatus::kReady);
      device_cache_.insert(new_tensor_info);
    }
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "CheckInputsAvailable end" << kSplitStr << ", current idx:" << idx;
}

KernelTensorPtrPairList MemCountedCache::CheckOutputsEnough(
  size_t idx, const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors,
  const device::DeviceContext *device_context) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "CheckOutputsEnough begin" << kSplitStr << ", current idx:" << idx;
  PrintCache(device_cache_, kDevice);
  PrintCache(host_cache_, kHost);
  size_t total_output_size = 0;
  KernelTensorPtrPairList ret;
  size_t stream_id = 0;
  for (const auto &kernel_tensor_info : kernel_tensors) {
    size_t cur_output_size = kernel_tensor_info.first->GetSize();
    total_output_size += cur_output_size;
    stream_id = kernel_tensor_info.first->stream_id();
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "total output size" << kSplitStr << ": " << total_output_size;
  size_t need_offload_size = GetNeedOffloadSize(total_output_size + buffer_size_);
  if (need_offload_size != 0) {
    ret = Offload(idx + 1, need_offload_size, stream_id, device_context);
    if (!ret.empty()) {
      MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Offload for check output, current idx: " << idx
                                     << " , move_out kernel number: " << ret.size()
                                     << " , need_offload_size: " << need_offload_size;
    }
    auto must_offload_size = GetNeedOffloadSize(total_output_size);
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Offload size: " << need_offload_size
                                   << " , must offload size: " << must_offload_size;
    InsertWaitForCopyOutData(device_context, must_offload_size);
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "CheckOutputsEnough end" << kSplitStr << ", current idx:" << idx;
  return ret;
}

void MemCountedCache::CleanExpiredDeviceTensors(size_t current_idx) {
  for (auto it = device_cache_.begin(); it != device_cache_.end();) {
    auto &kernel_tensor_info = *it;
    if (kernel_tensor_info->next_use_idx > current_idx) {
      break;
    }
    if ((kernel_tensor_info->next_use_idx == current_idx && kernel_tensor_info->need_free) ||
        kernel_tensor_info->next_use_idx < current_idx) {
      MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << "Clean expired data: " << kernel_tensor_info->kernel_tensor;
      it = device_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

void MemCountedCache::InsertNewTensorsToDevice(
  size_t idx, const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors_info) {
  for (auto &kernel_tensor_info : kernel_tensors_info) {
    TensorInfoPtr info =
      std::make_shared<TensorInfo>(kernel_tensor_info.first, kernel_tensor_info.second, false, TensorStatus::kReady);
    if (kernel_tensor_info.second == SIZE_MAX) {
      info->next_use_idx = idx;
      info->need_free = true;
    }
    device_cache_.insert(info);
  }
}

KernelTensorPtrPairList MemCountedCache::LoadAllWithinHorizon(size_t current_idx, size_t horizon, uint32_t stream_id,
                                                              const device::DeviceContext *device_context) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "LoadAllWithinHorizon begin" << kSplitStr
                                 << ", current idx:" << current_idx;
  RemoteActionPtrList host_to_device_actions;
  std::unordered_map<KernelTensorPtr, size_t> tensor_to_idx;
  for (const auto &kernel_tensor_info : host_cache_) {
    if (kernel_tensor_info->next_use_idx > current_idx + horizon) {
      break;
    }
    // Todo: no need free original tensor, refresh kernel tensor info, use original directly
    if (kernel_tensor_info->status == TensorStatus::kCopyingToHost) {
      MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "No need move out node, need check: "
                                        << kernel_tensor_info->kernel_tensor->ToString()
                                        << ", wait_idx: " << kernel_tensor_info->copy_out_wait_op_num
                                        << ", current_idx: " << current_idx;
      kernel_tensor_info->copy_out_wait_op_num = 0;
      kernel_tensor_info->status = TensorStatus::kReady;
      auto action = std::make_shared<RemoteAction>(HyperOffloadEventType::kWaitWithMemoryEvent,
                                                   kernel_tensor_info->old_kernel_tensor, to_host_stream_, stream_id);
      (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{action}, device_context);
    }

    if (kernel_tensor_info->status != TensorStatus::kReady) {
      continue;
    }
    auto action = std::make_shared<RemoteAction>(HyperOffloadEventType::kHostToDevice,
                                                 kernel_tensor_info->kernel_tensor, to_device_stream_);
    host_to_device_actions.emplace_back(action);
    tensor_to_idx.emplace(kernel_tensor_info->kernel_tensor, kernel_tensor_info->next_use_idx);
  }
  KernelTensorPtrPairList kernel_tensor_pair_list;
  if (!host_to_device_actions.empty()) {
    auto record_wait_action = std::make_shared<RemoteAction>(HyperOffloadEventType::kRecordWaitPairEvent, nullptr,
                                                             stream_id, to_device_stream_);
    host_to_device_actions.insert(host_to_device_actions.begin(), record_wait_action);
    kernel_tensor_pair_list = mem_action_mgr_->CreateRemoteEvents(host_to_device_actions, device_context);
    RemoteActionPtrList record_action_list;
    for (const auto &kernel_tensor_pair : kernel_tensor_pair_list) {
      if (kernel_tensor_pair.first == nullptr || kernel_tensor_pair.second == nullptr) {
        continue;
      }
      UpdateTensorStatus(kernel_tensor_pair.first, HyperOffloadEventType::kHostToDevice);
      auto record_action = std::make_shared<RemoteAction>(HyperOffloadEventType::kRecordWithMemoryEvent,
                                                          kernel_tensor_pair.first, to_device_stream_, stream_id);
      record_action_list.emplace_back(record_action);
      auto idx_iter = tensor_to_idx.find(kernel_tensor_pair.first);
      if (idx_iter == tensor_to_idx.end()) {
        MS_LOG(WARNING) << "Cannot find tensor index for: " << kernel_tensor_pair.first;
        continue;
      }
      TensorInfoPtr new_tensor_info = std::make_shared<TensorInfo>(
        kernel_tensor_pair.second, idx_iter->second, false, TensorStatus::kCopyingToDevice, kernel_tensor_pair.first);
      device_cache_.insert(new_tensor_info);
    }
    (void)mem_action_mgr_->CreateRemoteEvents(record_action_list, device_context);
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "LoadAllWithinHorizon end" << kSplitStr
                                 << ", current idx:" << current_idx;
  return kernel_tensor_pair_list;
}

KernelTensorPtrPairList MemCountedCache::ProcessOutput(
  const device::DeviceContext *device_context, uint32_t stream_id, size_t idx,
  const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors_info) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kSplitStr << "ProcessOutput begin" << kSplitStr << ", current idx:" << idx;
  PrintKernelTensorList(kernel_tensors_info, kOutput);
  InsertNewTensorsToDevice(idx, kernel_tensors_info);
  CleanExpiredDeviceTensors(idx);
  PrintCache(device_cache_, kDevice);
  PrintCache(host_cache_, kHost);

  cur_inputs_.clear();
  size_t horizon = GetActualHorizon(idx);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Cur idx: " << idx << " , horizon: " << horizon;

  size_t total_load_size = 0;
  for (const auto &kernel_tensor_info : host_cache_) {
    if (kernel_tensor_info->next_use_idx > idx + horizon) {
      break;
    }
    if (kernel_tensor_info->status != TensorStatus::kReady) {
      continue;
    }
    size_t cur_tensor_size = kernel_tensor_info->kernel_tensor->GetSize();
    total_load_size += cur_tensor_size;
  }

  size_t need_offload_size = GetNeedOffloadSize(total_load_size + buffer_size_);
  if (need_offload_size == 0) {
    return LoadAllWithinHorizon(idx, horizon, stream_id, device_context);
  }

  KernelTensorPtrPairList ret = Offload(idx + horizon + 1, need_offload_size, stream_id, device_context);
  if (!ret.empty()) {
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Offload for process output, current idx: " << idx
                                   << " , move_out kernel number: " << ret.size()
                                   << " , need_offload_size: " << need_offload_size;
  }
  auto must_offload_size = GetNeedOffloadSize(total_load_size);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Offload size: " << need_offload_size
                                 << " , must offload size: " << must_offload_size;
  InsertWaitForCopyOutData(device_context, must_offload_size, false);
  auto load_pairs = LoadAllWithinHorizon(idx, horizon, stream_id, device_context);
  ret.insert(ret.end(), load_pairs.begin(), load_pairs.end());
  return ret;
}

void MemCountedCache::UpdateTensorStatus(const KernelTensorPtr &kernel_tensor, HyperOffloadEventType event_type) {
  TensorInfoPtrSet *target_cache = nullptr;
  if (event_type == HyperOffloadEventType::kDeviceToHost) {
    target_cache = &device_cache_;
  } else if (event_type == HyperOffloadEventType::kHostToDevice) {
    target_cache = &host_cache_;
  } else {
    return;
  }
  MS_EXCEPTION_IF_NULL(target_cache);
  auto it = std::find_if(target_cache->begin(), target_cache->end(), [&kernel_tensor](const TensorInfoPtr &info) {
    return info->kernel_tensor.get() == kernel_tensor.get();
  });
  if (it != target_cache->end()) {
    target_cache->erase(it);
  }
}

}  // namespace runtime
}  // namespace mindspore
