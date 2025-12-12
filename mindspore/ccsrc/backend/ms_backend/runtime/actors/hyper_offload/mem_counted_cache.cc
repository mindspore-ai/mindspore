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

namespace {
void PrintCache(const TensorInfoPtrSet &cache) {
  for (const auto &cc : cache) {
    MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << cc->kernel_tensor << " " << cc->status << " " << cc->next_use_idx;
  }
}
}  // namespace

void MemCountedCache::SetCopyStreamId(size_t copy_out_stream_id, size_t copy_in_stream_id) {
  to_host_stream_ = copy_out_stream_id;
  to_device_stream_ = copy_in_stream_id;
}

size_t MemCountedCache::GetActualHorizon(size_t cur_idx) {
  if (cur_idx == next_conditionswitch_idx_ || cur_idx + horizon_ <= next_conditionswitch_idx_ ||
      conditionswitch_idxs_.empty()) {
    return horizon_;
  }
  if (cur_idx > next_conditionswitch_idx_) {
    next_conditionswitch_idx_ = conditionswitch_idxs_.front();
    conditionswitch_idxs_.pop();
    return GetActualHorizon(cur_idx);
  }
  return next_conditionswitch_idx_ - cur_idx;
}

TensorInfoPtrList MemCountedCache::GetCanOffloadTensorInfo(size_t first_offload_idx) {
  TensorInfoPtrList ret;
  for (auto it = device_cache_.rbegin(); it != device_cache_.rend(); ++it) {
    const auto &ptr = *it;
    if (ptr->next_use_idx < first_offload_idx) {
      break;
    }
    if (ptr->status != TensorStatus::kReady) {
      continue;
    }
    if (std::find(cur_inputs_.begin(), cur_inputs_.end(), ptr->kernel_tensor) != cur_inputs_.end()) {
      continue;
    }
    ret.emplace_back(ptr);
  }
  return ret;
}

KernelTensorPtrPairList MemCountedCache::Offload(size_t first_offloadable_idx, size_t need_size, size_t stream_id,
                                                 const device::DeviceContext *device_context) {
  KernelTensorPtrPairList ret;
  auto can_offload_tensors = GetCanOffloadTensorInfo(first_offloadable_idx);
  if (can_offload_tensors.empty()) {
    return ret;
  }

  size_t has_free_size = 0;
  auto record_wait_before_action =
    std::make_shared<RemoteAction>(HyperOffloadEventType::kRecordWaitPairEvent, nullptr, stream_id, to_host_stream_);
  (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{record_wait_before_action}, device_context);
  for (auto it : can_offload_tensors) {
    auto action = std::make_shared<RemoteAction>(HyperOffloadEventType::kDeviceToHost, it->kernel_tensor, to_host_stream_);
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
    auto wait_after_action = std::make_shared<RemoteAction>(HyperOffloadEventType::kWaitWithMemoryEvent,
                                                            copy_kernel_tensor_pair.first, to_host_stream_, stream_id);
    (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{record_after_action, wait_after_action},
                                              device_context);
    ret.emplace_back(copy_kernel_tensor_pair);
    auto new_info = std::make_shared<TensorInfo>(copy_kernel_tensor_pair.second, it->next_use_idx, false,
                                                 TensorStatus::kReady, copy_kernel_tensor_pair.first);
    host_cache_.insert(new_info);
    UpdateTensorStatus(copy_kernel_tensor_pair.first, HyperOffloadEventType::kDeviceToHost);
    has_free_size += it->kernel_tensor->size();
    if (has_free_size >= need_size) {
      break;
    }
  }
  return ret;
}

size_t MemCountedCache::GetIdleMemSize() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey device_key = {device::DeviceType::kAscend, device_id};
  device::DeviceContext *device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_key);
  size_t total_mem = device_context->device_res_manager_->GetTotalMemStatistics();
  size_t used_mem = device_context->device_res_manager_->GetTotalUsedMemStatistics();
  if (total_mem < used_mem) {
    MS_EXCEPTION(RuntimeError) << "Memory not enough, total mem less than used mem, total mem: " << total_mem
                               << ", used mem: " << used_mem;
  }
  size_t idle_mem = total_mem - used_mem;
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************idle_mem***************: " << idle_mem;

  return idle_mem;
}

size_t MemCountedCache::GetDeviceAvailableMemSize() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey device_key = {device::DeviceType::kAscend, device_id};
  device::DeviceContext *device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_key);
  size_t total_mem = device_context->device_res_manager_->GetTotalMemStatistics();
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************total_mem***************: " << total_mem;
  if (max_mem_ < total_mem) {
    MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Device memory not enough.";
    return 0;
  }

  return max_mem_ - total_mem;
}

size_t MemCountedCache::GetUsedMemSize() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey device_key = {device::DeviceType::kAscend, device_id};
  device::DeviceContext *device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_key);
  size_t used_mem = device_context->device_res_manager_->GetTotalUsedMemStatistics();

  return used_mem;
}

void MemCountedCache::CheckInputsAvailable(size_t idx,
                                           const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors,
                                           const device::DeviceContext *device_context) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************CheckInputsAvailable begin***************" << idx;
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************print inputs begin***************";
  for (auto &kk : kernel_tensors) {
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << kk.first;
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************print inputs end***************";
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print device cache begin***************";
  PrintCache(device_cache_);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print device cache end***************";
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print host cache begin***************";
  PrintCache(host_cache_);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print host cache end***************";
  for (const auto &kernel_tensor_info : kernel_tensors) {
    cur_inputs_.emplace_back(kernel_tensor_info.first);
    KernelTensorPtr kernel_tensor = kernel_tensor_info.first;
    size_t next_use_idx = kernel_tensor_info.second;
    auto it = std::find_if(device_cache_.begin(), device_cache_.end(), [&kernel_tensor](const TensorInfoPtr &info) {
      return info->kernel_tensor.get() == kernel_tensor.get();
    });
    auto it_host = std::find_if(host_cache_.begin(), host_cache_.end(), [&kernel_tensor](const TensorInfoPtr &info) {
      return info->kernel_tensor.get() == kernel_tensor.get();
    });
    if (it != device_cache_.end()) {
      auto ptr = *it;
      if (ptr->status == TensorStatus::kReady) {
        TensorInfoPtr new_tensor_info =
          std::make_shared<TensorInfo>(kernel_tensor, next_use_idx, false, TensorStatus::kReady);
        if (next_use_idx == SIZE_MAX) {
          new_tensor_info->next_use_idx = idx;
          new_tensor_info->need_free = true;
        }
        device_cache_.erase(it);
        device_cache_.insert(new_tensor_info);
      } else if (ptr->status == TensorStatus::kCopyingToDevice) {
        MS_EXCEPTION_IF_NULL(ptr->old_kernel_tensor);
        auto action = std::make_shared<RemoteAction>(HyperOffloadEventType::kWaitWithMemoryEvent, ptr->old_kernel_tensor,
                                                     to_device_stream_, kernel_tensor->stream_id());
        (void)mem_action_mgr_->CreateRemoteEvents(RemoteActionPtrList{action}, device_context);
        TensorInfoPtr new_tensor_info =
          std::make_shared<TensorInfo>(kernel_tensor, next_use_idx, false, TensorStatus::kReady);
        if (next_use_idx == SIZE_MAX) {
          new_tensor_info->next_use_idx = idx;
          new_tensor_info->need_free = true;
        }
        device_cache_.erase(it);
        device_cache_.insert(new_tensor_info);
      }
    } else if (it_host != host_cache_.end()) {
      MS_EXCEPTION(RuntimeError) << "CopyToDevice Error: can't find kernel tensor in device_cache, memory not enough.";
    } else {
      // Insert target kernel tensor to device cache
      TensorInfoPtr new_info = std::make_shared<TensorInfo>(kernel_tensor, next_use_idx, false, TensorStatus::kReady);
      if (next_use_idx == SIZE_MAX) {
        new_info->next_use_idx = idx;
        new_info->need_free = true;
      }
      device_cache_.insert(new_info);
    }
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************CheckInputsAvailable end***************";
}

KernelTensorPtrPairList MemCountedCache::CheckOutputsEnough(
  size_t idx, const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors,
  const device::DeviceContext *device_context) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************CheckOutputsEnough***************" << idx;
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print device cache begin***************";
  PrintCache(device_cache_);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print device cache end***************";
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print host cache begin***************";
  PrintCache(host_cache_);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print host cache end***************";
  size_t device_available_size = GetDeviceAvailableMemSize();
  size_t idle_size = GetIdleMemSize();
  size_t total_output_size = 0;
  KernelTensorPtrPairList ret;
  size_t stream_id = 0;
  for (const auto &kernel_tensor_info : kernel_tensors) {
    size_t cur_output_size = kernel_tensor_info.first->GetSize();
    total_output_size += cur_output_size;
    stream_id = kernel_tensor_info.first->stream_id();
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************total output size***************: " << total_output_size;
  if (device_available_size >= total_output_size || idle_size >= total_output_size) {
    return ret;
  }
  ret = Offload(idx + 1, total_output_size, stream_id, device_context);

  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************CheckOutputsEnough end***************";
  return ret;
}

void MemCountedCache::CleanExpiredDeviceTensors(size_t current_idx) {
  for (auto it = device_cache_.begin(); it != device_cache_.end();) {
    auto ptr = *it;
    if (ptr->next_use_idx > current_idx) {
      break;
    }
    if ((ptr->next_use_idx == current_idx && ptr->need_free) || ptr->next_use_idx < current_idx) {
      MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << "Clean expired data: " << ptr->kernel_tensor;
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
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************LoadAllWithinHorizon begin***************";
  RemoteActionPtrList host_to_device_actions;
  std::unordered_map<KernelTensorPtr, size_t> tensor_to_idx;
  for (auto it = host_cache_.begin(); it != host_cache_.end(); ++it) {
    auto ptr = *it;
    if (ptr->next_use_idx > current_idx + horizon) {
      break;
    }
    if (ptr->status != TensorStatus::kReady) {
      continue;
    }
    auto action =
      std::make_shared<RemoteAction>(HyperOffloadEventType::kHostToDevice, ptr->kernel_tensor, to_device_stream_);
    host_to_device_actions.emplace_back(action);
    tensor_to_idx.emplace(ptr->kernel_tensor, ptr->next_use_idx);
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
      TensorInfoPtr new_tensor_info =
        std::make_shared<TensorInfo>(kernel_tensor_pair.second, tensor_to_idx[kernel_tensor_pair.first], false,
                                     TensorStatus::kCopyingToDevice, kernel_tensor_pair.first);
      device_cache_.insert(new_tensor_info);
    }
    (void)mem_action_mgr_->CreateRemoteEvents(record_action_list, device_context);
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************LoadAllWithinHorizon end***************";
  return kernel_tensor_pair_list;
}

KernelTensorPtrPairList MemCountedCache::ProcessOutput(
  const device::DeviceContext *device_context, uint32_t stream_id, size_t idx,
  const std::vector<std::pair<KernelTensorPtr, size_t>> &kernel_tensors_info) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************ProcessOutput***************" << idx;
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************print output begin***************";
  for (const auto &oo : kernel_tensors_info) {
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << oo.first;
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************print output end***************";
  InsertNewTensorsToDevice(idx, kernel_tensors_info);
  CleanExpiredDeviceTensors(idx);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print device cache begin***************";
  PrintCache(device_cache_);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print device cache end***************";
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print host cache begin***************";
  PrintCache(host_cache_);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "***************Print host cache end***************";

  // clear
  cur_inputs_.clear();
  size_t horizon = GetActualHorizon(idx);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Cur idx: " << idx << " , horizon: " << horizon;

  size_t total_load_size = 0;
  for (auto it : host_cache_) {
    if (it->next_use_idx > idx + horizon) {
      break;
    }
    if (it->status != TensorStatus::kReady) {
      continue;
    }
    size_t cur_tensor_size = it->kernel_tensor->GetSize();
    total_load_size += cur_tensor_size;
  }

  size_t device_available_size = GetDeviceAvailableMemSize();
  size_t idle_size = GetIdleMemSize();
  total_load_size += buffer_size_;
  if (device_available_size >= total_load_size || idle_size >= total_load_size) {
    return LoadAllWithinHorizon(idx, horizon, stream_id, device_context);
  }

  KernelTensorPtrPairList ret = Offload(idx + horizon + 1, total_load_size, stream_id, device_context);
  auto load_pairs = LoadAllWithinHorizon(idx, horizon, stream_id, device_context);
  ret.insert(ret.end(), load_pairs.begin(), load_pairs.end());
  return ret;
}

void MemCountedCache::UpdateTensorStatus(const KernelTensorPtr &kernel_tensor, HyperOffloadEventType event_type) {
  if (event_type == HyperOffloadEventType::kDeviceToHost) {
    auto it = std::find_if(device_cache_.begin(), device_cache_.end(), [&kernel_tensor](const TensorInfoPtr &info) {
      return info->kernel_tensor.get() == kernel_tensor.get();
    });
    if (it != device_cache_.end()) {
      device_cache_.erase(it);
    }
  }

  if (event_type == HyperOffloadEventType::kHostToDevice) {
    auto it = std::find_if(host_cache_.begin(), host_cache_.end(), [&kernel_tensor](const TensorInfoPtr &info) {
      return info->kernel_tensor.get() == kernel_tensor.get();
    });
    if (it != host_cache_.end()) {
      host_cache_.erase(it);
    }
  }
}

}  // namespace runtime
}  // namespace mindspore
