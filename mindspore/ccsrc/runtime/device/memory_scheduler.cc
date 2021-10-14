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
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
void MemScheduler::Clear() {
  if (mem_handler_ == nullptr) {
    return;
  }
  for (auto &item : high_priority_device_ptr_) {
    mem_handler_->FreeDevice(item.second);
  }
  high_priority_device_ptr_.clear();
}

bool MemScheduler::IsHighPriorityMem(const void *key) {
  auto iter = mem_priority_.find(key);
  if (iter != mem_priority_.end()) {
    return iter->second == kMemPriorityHigh;
  }
  return false;
}

void MemScheduler::SetMemPriority(const void *key, MemPriority priority) { mem_priority_[key] = priority; }

void MemScheduler::Record(const void *key, const EventType &event_type, size_t mem_size) {
  if (key == nullptr) {
    return;
  }
  auto event = std::make_shared<Event>(event_type, compute_index_);
  event->mem_size = mem_size;
  event->key = key;
  (void)mem_events_[key].emplace_back(event);
}

void MemScheduler::Init(const void *key, void *host_ptr, size_t mem_size, MemPriority priority) {
  if (need_record_event_) {
    mem_priority_[key] = priority;
    Record(key, kInit, mem_size);
  } else {
    init_host_ptr_[key] = host_ptr;
  }
}

void *MemScheduler::GetOrMalloc(const void *key, size_t mem_size, MemPriority priority) {
  if (need_record_event_) {
    if (mem_priority_.find(key) == mem_priority_.end()) {
      mem_priority_[key] = priority;
      Record(key, kMalloc, mem_size);
    } else {
      Record(key, kGet, mem_size);
    }
    return nullptr;
  }
  auto iter = mem_result_.find(key);
  if (iter != mem_result_.end()) {
    auto ptr = iter->second;
    MS_EXCEPTION_IF_NULL(ptr);
    return ptr;
  } else {
    MS_LOG_EXCEPTION << "Mem extender get nullptr result!";
  }
}

bool MemScheduler::PreCompute(void *stream) {
  if (need_record_event_) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(mem_handler_);
  if (pre_compute_events_.size() <= compute_index_) {
    MS_LOG_EXCEPTION << "Index out of pre event range, index:" << compute_index_
                     << ", event size:" << pre_compute_events_.size();
  }
  auto &events = pre_compute_events_[compute_index_];
  for (auto &event : events) {
    MS_EXCEPTION_IF_NULL(event);
    MS_LOG(DEBUG) << "Pre compute " << compute_index_ << ": " << event->key << " v " << event->type;
    if (event->type == kInit) {
      auto host_ptr = init_host_ptr_[event->key];
      MS_EXCEPTION_IF_NULL(host_ptr);
      auto priority = mem_priority_[event->key];
      auto iter = high_priority_device_ptr_.find(event->key);
      if (priority != kMemPriorityLow && iter != high_priority_device_ptr_.end()) {
        MS_EXCEPTION_IF_NULL(iter->second);
        mem_result_[event->key] = iter->second;
        if (priority == kMemPriorityMedium) {
          mem_handler_->SwapIn(host_ptr, iter->second, event->mem_size, stream);
        }
        continue;
      }
      auto device_ptr = mem_handler_->MallocDevice(event->mem_size);
      if (device_ptr == nullptr) {
        return false;
      }
      if (priority != kMemPriorityLow) {
        high_priority_device_ptr_[event->key] = device_ptr;
      }
      mem_handler_->SwapIn(host_ptr, device_ptr, event->mem_size, stream);
      mem_result_[event->key] = device_ptr;
    } else if (event->type == kMalloc) {
      auto device_ptr = mem_handler_->MallocDevice(event->mem_size);
      if (device_ptr == nullptr) {
        return false;
      }
      mem_result_[event->key] = device_ptr;
    } else if (event->type == kSwapIn) {
      bool from_init = true;
      auto host_ptr = init_host_ptr_[event->key];
      if (host_ptr == nullptr) {
        host_ptr = swap_host_ptr_[event->key];
        from_init = false;
      }
      auto device_ptr = mem_handler_->MallocDevice(event->mem_size);
      if (device_ptr == nullptr) {
        return false;
      }
      MS_EXCEPTION_IF_NULL(host_ptr);
      mem_handler_->SwapIn(host_ptr, device_ptr, event->mem_size, stream);
      mem_result_[event->key] = device_ptr;
      if (!from_init) {
        mem_handler_->FreeHost(host_ptr);
        (void)swap_host_ptr_.erase(event->key);
      }
    }
  }
  return true;
}

bool MemScheduler::PostCompute(void *stream) {
  if (need_record_event_) {
    ++compute_index_;
    return true;
  }
  MS_EXCEPTION_IF_NULL(mem_handler_);
  if (post_compute_events_.size() <= compute_index_) {
    MS_LOG_EXCEPTION << "Index out of post event range, index:" << compute_index_
                     << ", event size:" << post_compute_events_.size();
  }
  auto &events = post_compute_events_[compute_index_];
  for (auto &event : events) {
    MS_EXCEPTION_IF_NULL(event);
    MS_LOG(DEBUG) << "Post compute " << compute_index_ << ": " << event->key << " v " << event->type;
    if (event->type == kFree) {
      auto ptr = mem_result_[event->key];
      if (ptr == nullptr) {
        return false;
      }
      mem_handler_->FreeDevice(ptr);
      (void)mem_result_.erase(event->key);
    } else if (event->type == kSwapOut) {
      auto device_ptr = mem_result_[event->key];
      if (device_ptr == nullptr) {
        return false;
      }
      auto host_ptr = init_host_ptr_[event->key];
      if (host_ptr == nullptr) {
        host_ptr = mem_handler_->MallocHost(event->mem_size);
        swap_host_ptr_[event->key] = host_ptr;
      }
      MS_EXCEPTION_IF_NULL(host_ptr);
      mem_handler_->SwapOut(device_ptr, host_ptr, event->mem_size, stream);
      mem_handler_->FreeDevice(device_ptr);
      (void)mem_result_.erase(device_ptr);
    }
  }
  ++compute_index_;
  return true;
}

void MemScheduler::OptMemUsage() {
  need_record_event_ = false;
  if (optimized_) {
    return;
  }
  CountMemUsage();
  CheckMemSize();
  if (need_swap_) {
    GenEventSpan();
    GenNoSwapEventSet();
  }
  GenEvents();
}

void MemScheduler::CountMemUsage() {
  if (!min_mem_used_.empty()) {
    return;
  }
  min_mem_used_.resize(compute_index_, 0);
  std::vector<size_t> total_mem_used(compute_index_, 0);
  for (auto &item : mem_events_) {
    auto &mem_events = item.second;
    if (mem_events.empty()) {
      continue;
    }
    auto first_event = mem_events[0];
    MS_EXCEPTION_IF_NULL(first_event);
    size_t i = 0;
    if (first_event->type == kInit && mem_events.size() > 1) {
      first_event = mem_events[1];
      i = 1;
    }
    auto last_event = mem_events[mem_events.size() - 1];
    for (size_t start_index = first_event->index; start_index <= last_event->index; ++start_index) {
      if (start_index < compute_index_) {
        total_mem_used[start_index] += first_event->mem_size;
      } else {
        MS_LOG(ERROR) << "Error mem event index " << start_index;
      }
    }
    for (; i < mem_events.size(); ++i) {
      auto &event = mem_events[i];
      MS_EXCEPTION_IF_NULL(event);
      if (event->index < compute_index_) {
        min_mem_used_[event->index] += first_event->mem_size;
      } else {
        MS_LOG(ERROR) << "Error mem event index " << event->index;
      }
    }
  }
  min_mem_needed_ = *(std::max_element(min_mem_used_.begin(), min_mem_used_.end()));
  mem_used_without_swap_ = *(std::max_element(total_mem_used.begin(), total_mem_used.end()));
}

void MemScheduler::CheckMemSize() {
  MS_EXCEPTION_IF_NULL(mem_handler_);
  auto available_mem_size = mem_handler_->GetAvailableMemSize();
  if (available_mem_size < min_mem_needed_) {
    MS_LOG(EXCEPTION) << "Out of memory, as available mem size is " << available_mem_size
                      << " while graph needs at least " << min_mem_needed_;
  }
  if (mem_used_without_swap_ > available_mem_size) {
    need_swap_ = true;
  }
  MS_LOG(INFO) << "Available mem size: " << available_mem_size << ", graph needs mem size:" << mem_used_without_swap_
               << "without swap, and needs at least " << min_mem_needed_ << " with swap.";
}

void MemScheduler::GenEventSpan() {
  if (!event_span_.empty()) {
    return;
  }
  for (auto &item : mem_events_) {
    auto &mem_events = item.second;
    if (mem_events.empty()) {
      continue;
    }
    auto first_event = mem_events[0];
    MS_EXCEPTION_IF_NULL(first_event);
    size_t i = 0;
    if (first_event->type == kInit && mem_events.size() > 1) {
      first_event = mem_events[1];
      i = 1;
    }
    size_t last_index = first_event->index;
    for (; i < mem_events.size(); ++i) {
      auto &event = mem_events[i];
      MS_EXCEPTION_IF_NULL(event);
      auto span = event->index - last_index;
      if (span > 1) {
        (void)event_span_.emplace(std::pair<size_t, std::shared_ptr<Event>>(span, event));
      }
      last_index = event->index;
    }
  }
}

void MemScheduler::GenNoSwapEventSet() {
  MS_EXCEPTION_IF_NULL(mem_handler_);
  auto available_mem_size = mem_handler_->GetAvailableMemSize();
  auto threshold = available_mem_size * mem_used_factor_;
  no_swap_events_.clear();
  std::vector<size_t> cur_mem_used(min_mem_used_.begin(), min_mem_used_.end());
  for (auto iter = event_span_.begin(); iter != event_span_.end(); ++iter) {
    auto span = iter->first;
    auto &event = iter->second;
    auto start_index = event->index - span + 1;
    bool revert = false;
    for (size_t i = start_index; i < event->index; ++i) {
      cur_mem_used[i] += event->mem_size;
      if (cur_mem_used[i] > threshold) {
        revert = true;
      }
    }
    if (revert) {
      for (size_t i = start_index; i < event->index; ++i) {
        cur_mem_used[i] -= event->mem_size;
      }
    } else {
      (void)no_swap_events_.emplace(event);
    }
  }
}

void MemScheduler::GenEvents() {
  pre_compute_events_.resize(compute_index_);
  post_compute_events_.resize(compute_index_);
  for (auto &item : mem_events_) {
    auto &mem_events = item.second;
    if (mem_events.empty()) {
      continue;
    }
    auto first_event = mem_events[0];
    MS_EXCEPTION_IF_NULL(first_event);
    if (first_event->type == kInit) {
      if (mem_events.size() > 1) {
        auto &second_event = mem_events[1];
        MS_EXCEPTION_IF_NULL(second_event);
        first_event->index = second_event->index;
      } else {
        continue;
      }
    }
    if ((first_event->type == kInit || first_event->type == kMalloc) &&
        first_event->index < pre_compute_events_.size()) {
      (void)pre_compute_events_[first_event->index].emplace_back(first_event);
    } else {
      MS_LOG_EXCEPTION << "First event should be init or malloc!";
    }
    MemPriority priority = kMemPriorityLow;
    auto iter = mem_priority_.find(first_event->key);
    if (iter != mem_priority_.end()) {
      priority = iter->second;
    }
    size_t pre_index = first_event->index;
    for (size_t i = 1; i < mem_events.size(); ++i) {
      auto &event = mem_events[i];
      MS_EXCEPTION_IF_NULL(event);
      if (need_swap_ && event->index - pre_index > 1 && priority == kMemPriorityLow &&
          no_swap_events_.find(event) == no_swap_events_.end()) {
        auto swap_out_event = std::make_shared<Event>(kSwapOut, pre_index);
        swap_out_event->key = item.first;
        swap_out_event->mem_size = first_event->mem_size;
        (void)post_compute_events_[pre_index].emplace_back(swap_out_event);
        auto swap_in_event = std::make_shared<Event>(kSwapIn, event->index);
        swap_in_event->key = item.first;
        swap_in_event->mem_size = first_event->mem_size;
        (void)pre_compute_events_[event->index].emplace_back(swap_in_event);
      }
      if (event->index < pre_compute_events_.size()) {
        (void)pre_compute_events_[event->index].emplace_back(event);
      }
      pre_index = event->index;
    }
    if (priority != kMemPriorityLow) {
      continue;
    }
    auto &last_event = mem_events[mem_events.size() - 1];
    MS_EXCEPTION_IF_NULL(last_event);
    auto free_event = std::make_shared<Event>(kFree, last_event->index);
    free_event->key = item.first;
    if (last_event->index < post_compute_events_.size()) {
      (void)post_compute_events_[last_event->index].emplace_back(free_event);
    }
  }
}
}  // namespace device
}  // namespace mindspore
