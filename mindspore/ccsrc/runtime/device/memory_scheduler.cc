/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/device/memory_scheduler.h"
#include <algorithm>
#include <queue>
#include <set>
#ifdef _MSC_VER
#include <time.h>
#else
#include <sys/time.h>
#endif
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace device {
namespace {
constexpr float kMaxMemReuseFactor = 1.0;
constexpr float kMinMemReuseFactor = 0.5;
constexpr float kRetryFactor = 0.1;
constexpr size_t kMockTimes = 5;

double GetCurrentTime() {
#ifdef _MSC_VER
  return time(NULL) * 1.0e6;
#else
  struct timeval tv;
  (void)gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1.0e6 + tv.tv_usec;
#endif
}
}  // namespace

void *MemHandler::MallocHost(size_t mem_size) {
  auto &mem_que = cached_host_mem_[mem_size];
  if (!mem_que.empty()) {
    auto ret = mem_que.front();
    mem_que.pop();
    return ret;
  }
  auto block = std::make_shared<std::vector<uint8_t>>();
  try {
    block->resize(mem_size, 0);
    auto ptr = block->data();
    host_mem_block_map_[ptr] = block;
    return ptr;
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Malloc memory failed: size " << mem_size;
  }
}

void MemHandler::FreeHost(void *ptr) {
  MS_EXCEPTION_IF_NULL(ptr);
  auto iter = host_mem_block_map_.find(ptr);
  if (iter == host_mem_block_map_.end()) {
    MS_LOG(ERROR) << "Free ptr not be created from manager!";
  }
  auto mem_size = iter->second->size();
  cached_host_mem_[mem_size].emplace(iter->first);
}

void MemScheduler::Clear() {
  if (mem_handler_ == nullptr) {
    return;
  }
  for (auto &item : mem_result_) {
    mem_handler_->FreeDevice(item.second);
  }
  mem_result_.clear();
}

void MemScheduler::ClearAllocatedMem() {
  if (mem_handler_ == nullptr) {
    return;
  }
  for (auto &item : mem_result_) {
    const auto device_ptr = item.second;
    if (device_ptr != nullptr) {
      mem_handler_->FreeDevice(device_ptr);
    }
  }
  mem_result_.clear();
  for (const auto &item : swap_host_ptr_) {
    const auto host_ptr = item.second;
    if (host_ptr != nullptr) {
      mem_handler_->FreeHost(host_ptr);
    }
  }
  swap_host_ptr_.clear();
  continuous_mem_key_.clear();
}

void MemScheduler::AddContinuousMemInfo(bool is_input, size_t compute_index, size_t total_size,
                                        const std::vector<size_t> &align_size_list,
                                        const std::vector<const void *> &address_key_list) {
  MS_EXCEPTION_IF_NULL(continuous_mem_info_helper_);
  continuous_mem_info_helper_->AddContinuousMemInfo(is_input, compute_index, total_size, align_size_list,
                                                    address_key_list);
}

void MemScheduler::Record(const void *key, const MemEventType &event_type, size_t mem_size) {
  if (key == nullptr) {
    return;
  }
  auto event = std::make_shared<MemEvent>(event_type, current_step_);
  event->mem_size = mem_size;
  event->key = key;
  (void)mem_events_[key].emplace_back(event);
  if (step_keys_.size() < current_step_ + 1) {
    step_keys_.resize(current_step_ + 1);
  }
  if (event->type == kGet) {
    (void)step_keys_[current_step_].insert(event->key);
  }
}

void MemScheduler::Init(const void *key, void *host_ptr, size_t mem_size, MemPriority priority) {
  if (need_record_event_) {
    mem_priority_[key] = priority;
    Record(key, kInit, mem_size);
  }
  init_host_ptr_[key] = host_ptr;
}

void *MemScheduler::GetOrMalloc(const void *key, size_t mem_size, MemPriority priority) {
  if (need_record_event_) {
    if (mem_priority_.find(key) == mem_priority_.end()) {
      mem_priority_[key] = priority;
      Record(key, kMalloc, mem_size);
    }
    Record(key, kGet, mem_size);
    return nullptr;
  }
  if (strategy_ == nullptr) {
    return nullptr;
  }
  auto iter = mem_result_.find(key);
  if (iter != mem_result_.end()) {
    auto ptr = iter->second;
    MS_EXCEPTION_IF_NULL(ptr);
    return ptr;
  }
  return nullptr;
}

void *MemScheduler::MallocContinuousMem(const std::shared_ptr<MemEvent> &event, void *stream) {
  const auto &continuous_mem_info = continuous_mem_info_helper_->GetContinuousMemInfo(event->key);
  void *device_ptr = nullptr;
  if (cur_step_allocated_continuous_mem_.count(continuous_mem_info) == 0 &&
      continuous_mem_info_helper_->NeedMallocContinuousMem(continuous_mem_info, current_step_)) {
    if (mem_result_.find(event->key) != mem_result_.end()) {
      MS_LOG(EXCEPTION) << "Device memory is allocated before first continuous memory alloc event, event key: "
                        << event->key << ", continuous memory used index: " << continuous_mem_info->compute_index_;
    }
    const auto &device_ptr_list =
      MallocContinuousMem(continuous_mem_info->total_size_, continuous_mem_info->align_size_list_, stream);
    if (device_ptr_list.empty()) {
      MS_LOG(WARNING) << "MallocContinuousMemFromMemPool failed, size: " << continuous_mem_info->total_size_;
      return nullptr;
    }
    for (const auto &key_index : continuous_mem_info->key_index_map_) {
      MS_EXCEPTION_IF_NULL(device_ptr_list[key_index.second]);
      mem_result_[key_index.first] = device_ptr_list[key_index.second];
      (void)continuous_mem_key_.insert(key_index.first);
    }
    device_ptr = mem_result_[event->key];
    MS_EXCEPTION_IF_NULL(device_ptr);
    (void)cur_step_allocated_continuous_mem_.insert(continuous_mem_info);
  } else {
    device_ptr = MallocDevice(event->mem_size, stream);
  }
  return device_ptr;
}

bool MemScheduler::PreComputeMock(const MemEventPtr &event) {
  const bool is_continuous_mem = continuous_mem_info_helper_->IsContinuousMem(event->key);
  void *device_ptr = nullptr;
  if (mem_result_.count(event->key) != 0) {
    return true;
  } else if (is_continuous_mem) {
    device_ptr = MallocContinuousMem(event, nullptr);
  } else {
    device_ptr = MallocDevice(event->mem_size, nullptr);
  }
  mem_result_[event->key] = device_ptr;
  return device_ptr != nullptr;
}

bool MemScheduler::PreComputeInit(const std::shared_ptr<MemEvent> &event, void *stream) {
  const bool is_continuous_mem = continuous_mem_info_helper_->IsContinuousMem(event->key);
  const auto &iter = mem_result_.find(event->key);
  const bool new_malloc = iter == mem_result_.end();
  void *device_ptr = nullptr;
  if (!new_malloc) {
    device_ptr = iter->second;
  } else if (is_continuous_mem) {
    device_ptr = MallocContinuousMem(event, stream);
  } else {
    device_ptr = MallocDevice(event->mem_size, stream);
  }
  if (device_ptr == nullptr) {
    return false;
  }

  if (new_malloc || high_priority_mem_need_init_.count(event->key) != 0) {
    MS_LOG(DEBUG) << "Init input data from host, key: " << event->key;
    auto host_ptr = init_host_ptr_[event->key];
    MS_EXCEPTION_IF_NULL(host_ptr);
    mem_handler_->SwapIn(host_ptr, device_ptr, event->mem_size, stream);
  }
  mem_result_[event->key] = device_ptr;
  return true;
}

bool MemScheduler::PreComputeMalloc(const std::shared_ptr<MemEvent> &event, void *stream) {
  const bool is_continuous_mem = continuous_mem_info_helper_->IsContinuousMem(event->key);
  void *device_ptr = nullptr;
  const auto &iter = mem_result_.find(event->key);
  if (iter != mem_result_.end()) {
    return true;
  } else if (is_continuous_mem) {
    device_ptr = MallocContinuousMem(event, stream);
  } else {
    device_ptr = MallocDevice(event->mem_size, stream);
  }
  if (device_ptr == nullptr) {
    return false;
  }
  mem_result_[event->key] = device_ptr;
  return true;
}

bool MemScheduler::PreComputeSwapIn(const std::shared_ptr<MemEvent> &event, void *stream) {
  if (!PreComputeMalloc(event, stream)) {
    return false;
  }
  PreComputeMalloc(event, stream);
  const auto device_ptr = mem_result_[event->key];
  MS_EXCEPTION_IF_NULL(device_ptr);
  bool from_init = true;
  void *host_ptr = nullptr;
  GetHostPtr(event->key, &host_ptr, &from_init);
  MS_EXCEPTION_IF_NULL(host_ptr);
  mem_handler_->SwapIn(host_ptr, device_ptr, event->mem_size, stream);
  mem_result_[event->key] = device_ptr;
  if (!from_init) {
    mem_handler_->FreeHost(host_ptr);
    (void)swap_host_ptr_.erase(event->key);
  }
  return true;
}

bool MemScheduler::PreComputeGet(const std::shared_ptr<MemEvent> &event, void *stream) {
  const auto key = event->key;
  const auto mem_size = event->mem_size;
  auto iter = mem_result_.find(key);
  if (iter != mem_result_.end()) {
    auto ptr = iter->second;
    MS_EXCEPTION_IF_NULL(ptr);
    return true;
  }
  if (!optimized_ || stream == nullptr) {
    return false;
  }
  void *host_ptr = nullptr;
  bool from_init = false;
  GetHostPtr(key, &host_ptr, &from_init);
  if (host_ptr == nullptr) {
    return false;
  }
  auto device_ptr = MallocDevice(mem_size, stream);
  mem_handler_->SwapIn(host_ptr, device_ptr, mem_size, stream);
  if (!from_init) {
    (void)swap_host_ptr_.erase(key);
    mem_handler_->FreeHost(host_ptr);
  }
  mem_result_[key] = device_ptr;
  return true;
}

bool MemScheduler::PreCompute(void *stream) {
  if (strategy_ == nullptr) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(mem_handler_);
  auto &events = strategy_->GetPreComputeEvents(current_step_);
  for (auto &event : events) {
    MS_EXCEPTION_IF_NULL(event);
    MS_LOG(DEBUG) << "Pre compute " << current_step_ << ": " << event->key << " v " << event->type;
    bool ret = true;
    if (!optimized_) {
      ret = PreComputeMock(event);
    } else if (event->type == kInit) {
      ret = PreComputeInit(event, stream);
    } else if (event->type == kMalloc) {
      ret = PreComputeMalloc(event, stream);
    } else if (event->type == kSwapIn) {
      ret = PreComputeSwapIn(event, stream);
    } else if (event->type == kGet) {
      ret = PreComputeGet(event, stream);
    }
    if (!ret) {
      cur_step_allocated_continuous_mem_.clear();
      return false;
    }
  }
  if (record_compute_time_ && !updated_) {
    compute_start_time_ = GetCurrentTime();
  }
  cur_step_allocated_continuous_mem_.clear();
  return true;
}

bool MemScheduler::PostCompute(void *stream) {
  if (strategy_ == nullptr) {
    ++current_step_;
    return true;
  }

  if (record_compute_time_ && !updated_ && current_step_ < compute_time_.size()) {
    compute_time_[current_step_] = GetCurrentTime() - compute_start_time_;
  }

  auto &events = strategy_->GetPostComputeEvents(current_step_);
  for (auto &event : events) {
    MS_EXCEPTION_IF_NULL(event);
    MS_LOG(DEBUG) << "Post compute " << current_step_ << ": " << event->key << " v " << event->type;
    if (!optimized_ || event->type == kFree) {
      auto ptr = mem_result_[event->key];
      if (ptr == nullptr) {
        return false;
      }
      mem_handler_->FreeDevice(ptr);
      (void)mem_result_.erase(event->key);
      continuous_mem_key_.erase(event->key);
    } else if (event->type == kSwapOut) {
      auto device_ptr = mem_result_[event->key];
      if (device_ptr == nullptr) {
        return false;
      }
      SwapOutAndFreeDevice(event->key, device_ptr, event->mem_size, stream);
    }
  }
  for (const auto &info : continuous_mem_info_helper_->GetIndexContinuousMemInfo(current_step_)) {
    for (const auto &key_index : info->key_index_map_) {
      continuous_mem_key_.erase(key_index.first);
    }
  }
  ++current_step_;
  return true;
}

void MemScheduler::OptMemUsage(float mem_used_factor) {
  MS_EXCEPTION_IF_NULL(mem_handler_);

  if (strategy_ == nullptr) {
    strategy_ = std::make_shared<MemOffloadStrategy>(mem_priority_, mem_events_, manual_offload_keys_, total_step_,
                                                     continuous_mem_info_helper_);
    if (manual_offload_keys_.empty()) {
      compute_time_.resize(total_step_);
    } else {
      updated_ = true;
    }
  }

  auto available_mem_size = mem_handler_->GetAvailableMemSize();
  available_mem_size = FloatToSize(available_mem_size * mem_used_factor);
  strategy_->set_mem_size(available_mem_size);
  strategy_->Execute();
}

bool MemScheduler::Optimize() {
  AdjustFirstEventIndex();
  float mem_used_factor = kMaxMemReuseFactor;
  while (mem_used_factor >= kMinMemReuseFactor) {
    bool ret = true;
    OptMemUsage(mem_used_factor);
    for (size_t mock_time = 0; mock_time < kMockTimes; ++mock_time) {
      ret = Mock();
      if (!ret) {
        break;
      }
    }
    if (ret) {
      optimized_ = true;
      return true;
    }
    ClearAllocatedMem();
    mem_used_factor -= kRetryFactor;
  }
  return false;
}

bool MemScheduler::Mock() {
  current_step_ = 0;
  for (size_t step = 0; step < total_step_; ++step) {
    bool ret = PreCompute(nullptr);
    if (!ret) {
      return false;
    }
    auto &step_keys = step_keys_[step];
    for (auto &key : step_keys) {
      auto ptr = GetOrMalloc(key, 0);
      if (ptr == nullptr) {
        return false;
      }
    }
    ret = PostCompute(nullptr);
    if (!ret) {
      return false;
    }
  }
  return true;
}

void MemScheduler::AdjustFirstEventIndex() {
  for (const auto &item : mem_events_) {
    const auto &mem_events = item.second;
    if (mem_events.empty()) {
      continue;
    }
    auto &first_event = mem_events[0];
    MS_EXCEPTION_IF_NULL(first_event);
    const auto &priority_iter = mem_priority_.find(item.first);
    const bool is_high_priority = (priority_iter != mem_priority_.end() && priority_iter->second == kMemPriorityHigh);
    if (first_event->type == kInit && !is_high_priority && mem_events.size() > 1) {
      const auto &second_event = mem_events[1];
      MS_EXCEPTION_IF_NULL(second_event);
      first_event->index = second_event->index;
    }
  }
}

void *MemScheduler::MallocDevice(size_t mem_size, void *stream) {
  const auto &no_reuse_key = step_keys_[current_step_];
  auto device_ptr = mem_handler_->MallocDevice(mem_size);
  if (device_ptr != nullptr || !optimized_) {
    return device_ptr;
  }
  // Find memory block big enough in mem_result_, except continuous mem and memory blocks used in this step.
  auto iter = mem_result_.begin();
  using KeySizePair = std::pair<const void *, size_t>;
  auto less = [](const KeySizePair &a, const KeySizePair &b) -> bool { return a.second < b.second; };
  std::priority_queue<KeySizePair, std::vector<KeySizePair>, decltype(less)> mem_can_swap(less);
  while (iter != mem_result_.end()) {
    const auto key = iter->first;
    if (no_reuse_key.count(key) != 0 || continuous_mem_key_.count(key) != 0) {
      ++iter;
      continue;
    }
    const auto device_mem_size = GetMemSize(key);
    if (device_mem_size >= mem_size) {
      SwapOutAndFreeDevice(key, iter->second, device_mem_size, stream);
      device_ptr = mem_handler_->MallocDevice(mem_size);
      MS_EXCEPTION_IF_NULL(device_ptr);
      return device_ptr;
    }
    mem_can_swap.push({key, device_mem_size});
    ++iter;
  }

  // Try swap out memory block from big to small
  while (!mem_can_swap.empty()) {
    const auto &max_mem_in_device = mem_can_swap.top();
    const auto key = max_mem_in_device.first;
    const auto swap_mem_size = max_mem_in_device.second;
    auto swap_device_ptr = mem_result_[key];
    MS_EXCEPTION_IF_NULL(swap_device_ptr);
    mem_can_swap.pop();
    SwapOutAndFreeDevice(key, swap_device_ptr, swap_mem_size, stream);
    device_ptr = mem_handler_->MallocDevice(mem_size);
    if (device_ptr != nullptr) {
      return device_ptr;
    }
  }

  return nullptr;
}

std::vector<void *> MemScheduler::MallocContinuousMem(size_t total_size, const std::vector<size_t> &size_list,
                                                      void *stream) {
  const auto &no_reuse_key = step_keys_[current_step_];
  auto device_ptr_list = mem_handler_->MallocContinuousMemFromMemPool(size_list);
  if (!device_ptr_list.empty() || !optimized_) {
    return device_ptr_list;
  }
  // Find memory block big enough in mem_result_, except continuous mem and memory blocks used in this step.
  auto iter = mem_result_.begin();
  using KeySizePair = std::pair<const void *, size_t>;
  auto less = [](const KeySizePair &a, const KeySizePair &b) -> bool { return a.second < b.second; };
  std::priority_queue<KeySizePair, std::vector<KeySizePair>, decltype(less)> mem_can_swap(less);
  while (iter != mem_result_.end()) {
    const auto key = iter->first;
    if (no_reuse_key.count(key) != 0 || continuous_mem_key_.count(key) != 0) {
      ++iter;
      continue;
    }
    const auto device_mem_size = GetMemSize(key);
    if (device_mem_size >= total_size) {
      SwapOutAndFreeDevice(key, iter->second, device_mem_size, stream);
      device_ptr_list = mem_handler_->MallocContinuousMemFromMemPool(size_list);
      if (device_ptr_list.empty()) {
        MS_LOG(EXCEPTION) << "device_ptr_list empty";
      }
      return device_ptr_list;
    }
    mem_can_swap.push({key, device_mem_size});
    ++iter;
  }

  // Try swap out memory block from big to small
  while (!mem_can_swap.empty()) {
    const auto &max_mem_in_device = mem_can_swap.top();
    mem_can_swap.pop();
    const auto key = max_mem_in_device.first;
    const auto swap_mem_size = max_mem_in_device.second;
    auto swap_device_ptr = mem_result_[key];
    MS_EXCEPTION_IF_NULL(swap_device_ptr);
    SwapOutAndFreeDevice(key, swap_device_ptr, swap_mem_size, stream);
    device_ptr_list = mem_handler_->MallocContinuousMemFromMemPool(size_list);
    if (!device_ptr_list.empty()) {
      return device_ptr_list;
    }
  }

  return device_ptr_list;
}

void MemScheduler::SwapOutAndFreeDevice(const void *key, void *device_ptr, size_t mem_size, void *stream) {
  void *host_ptr = nullptr;
  bool from_init = false;
  GetOrMallocHostPtr(key, mem_size, &host_ptr, &from_init);
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (!from_init || updated_high_priority_mem_.find(key) != updated_high_priority_mem_.end()) {
    mem_handler_->SwapOut(device_ptr, host_ptr, mem_size, stream);
    updated_high_priority_mem_.erase(key);
  }
  mem_handler_->FreeDevice(device_ptr);
  (void)mem_result_.erase(key);
}

size_t MemScheduler::GetMemSize(const void *key) {
  const auto &iter = mem_events_.find(key);
  if (iter == mem_events_.end() || iter->second.empty()) {
    MS_LOG(EXCEPTION) << "Get mem size for device address key[" << key << "] failed.";
  }
  return iter->second[0]->mem_size;
}

void MemScheduler::GetOrMallocHostPtr(const void *key, size_t mem_size, void **host_ptr, bool *from_init) {
  GetHostPtr(key, host_ptr, from_init);
  if (*host_ptr != nullptr) {
    return;
  }
  *host_ptr = mem_handler_->MallocHost(mem_size);
  MS_EXCEPTION_IF_NULL(*host_ptr);
  *from_init = false;
  swap_host_ptr_[key] = *host_ptr;
}

void MemScheduler::GetHostPtr(const void *key, void **host_ptr, bool *from_init) {
  auto iter = init_host_ptr_.find(key);
  if (iter != init_host_ptr_.end()) {
    *host_ptr = iter->second;
    *from_init = true;
    return;
  }
  iter = swap_host_ptr_.find(key);
  if (iter != swap_host_ptr_.end()) {
    *host_ptr = iter->second;
    *from_init = false;
    return;
  }
  *host_ptr = nullptr;
}

void MemScheduler::Update() {
  if (!optimized_) {
    return;
  }

  if (strategy_ == nullptr || !strategy_->need_swap()) {
    return;
  }

  if (updated_) {
    return;
  }

  if (!record_compute_time_) {
    record_compute_time_ = true;
    return;
  }

  strategy_->SetComputeTime(compute_time_);
  strategy_->Execute();
  updated_ = true;
}
}  // namespace device
}  // namespace mindspore
