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
#include "runtime/device/memory_offload_strategy.h"
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
constexpr size_t kFirstGetMemEventIndex = 1;
constexpr size_t kInitOrMallocMemEventIndex = 0;

std::vector<std::shared_ptr<MemEvent>> &MemOffloadStrategy::GetPreComputeEvents(size_t step) {
  if (pre_compute_events_.size() <= step) {
    MS_LOG_EXCEPTION << "Index out of pre event range, index:" << step << ", event size:" << pre_compute_events_.size();
  }
  return pre_compute_events_[step];
}

std::vector<std::shared_ptr<MemEvent>> &MemOffloadStrategy::GetPostComputeEvents(size_t step) {
  if (post_compute_events_.size() <= step) {
    MS_LOG_EXCEPTION << "Index out of post event range, index:" << step
                     << ", event size:" << post_compute_events_.size();
  }
  return post_compute_events_[step];
}

void MemOffloadStrategy::Execute() {
  CountMemUsage();
  CheckMemSize();
  if (need_swap_) {
    GenEventSpan();
    GenSwapEventSet();
  }
  GenComputeMemEvents();
}

void MemOffloadStrategy::CountMemUsage() {
  if (!min_mem_used_.empty()) {
    return;
  }
  if (mem_events_.empty() || total_step_ == 0) {
    return;
  }
  min_mem_used_.resize(total_step_, 0);
  std::vector<size_t> total_mem_used(total_step_, 0);
  size_t high_priority_mem_size = 0;
  for (auto &item : mem_events_) {
    auto &mem_events = item.second;
    if (mem_events.empty()) {
      continue;
    }
    auto first_event = mem_events[kInitOrMallocMemEventIndex];
    const bool is_high_priority = IsHighPriorityMem(first_event->key);
    if (is_high_priority) {
      high_priority_mem_size += first_event->mem_size;
    } else {
      auto last_event = mem_events[mem_events.size() - 1];
      for (size_t start_index = first_event->index; start_index <= last_event->index; ++start_index) {
        total_mem_used[start_index] += first_event->mem_size;
      }
    }
    // Calculate the minimum memory size for kernel execution.
    for (const auto &event : mem_events) {
      MS_EXCEPTION_IF_NULL(event);
      if (event->type != kGet) {
        continue;
      }
      min_mem_used_[event->index] += first_event->mem_size;
    }
  }
  min_mem_needed_ = *(std::max_element(min_mem_used_.begin(), min_mem_used_.end()));
  mem_used_without_swap_ = *(std::max_element(total_mem_used.begin(), total_mem_used.end())) + high_priority_mem_size;
  if (mem_size_ < min_mem_needed_) {
    MS_LOG(EXCEPTION) << "Out of memory, as available mem size is " << mem_size_ << " while graph needs at least "
                      << min_mem_needed_;
  }
}

bool MemOffloadStrategy::IsHighPriorityMem(const void *key) const {
  auto iter = mem_priority_.find(key);
  if (iter != mem_priority_.end()) {
    return iter->second == kMemPriorityHigh;
  }
  return false;
}

void MemOffloadStrategy::CheckMemSize() {
  if (mem_size_ < mem_used_without_swap_ || !manual_offload_keys_.empty()) {
    need_swap_ = true;
  }

  MS_LOG(INFO) << "Available mem size: " << mem_size_ << ", graph needs mem size: " << mem_used_without_swap_
               << " without swap, and needs at least " << min_mem_needed_ << " with swap.";
}

void MemOffloadStrategy::GenEventSpan() {
  if (!event_span_.empty()) {
    return;
  }
  for (auto &item : mem_events_) {
    auto &tensor_events = item.second;
    if (tensor_events.size() <= 1) {
      continue;
    }
    const bool is_high_priority = IsHighPriorityMem(tensor_events[kInitOrMallocMemEventIndex]->key);
    for (size_t i = kFirstGetMemEventIndex; i < tensor_events.size(); ++i) {
      auto &event = tensor_events[i];
      MS_EXCEPTION_IF_NULL(event);
      if (event->type != kGet) {
        MS_LOG(EXCEPTION) << "Event should be Get except fist event.";
      }
      auto latest_event = tensor_events[i - 1];
      if (i == kFirstGetMemEventIndex && is_high_priority) {
        latest_event = tensor_events[tensor_events.size() - 1];
      }
      auto span = GetSpanBetweenMemEvents(latest_event->index, event->index);
      if (is_high_priority && span == 0 && latest_event == event) {
        span = total_step_;
      }
      if (span > 1) {
        const size_t span_mul_size = (span - 1) * event->mem_size;
        (void)event_span_.emplace(std::make_pair(span_mul_size, std::make_pair(event, span)));
      }
    }
  }
}

void MemOffloadStrategy::GenSwapEventSet() {
  swap_events_.clear();
  // manual offload strategy
  if (!manual_offload_keys_.empty()) {
    for (const auto &iter : event_span_) {
      auto &event = iter.second.first;
      if (manual_offload_keys_.find(event->key) != manual_offload_keys_.end()) {
        (void)swap_events_.emplace(event);
      }
    }
    return;
  }

  // greedy span filter
  std::vector<size_t> cur_mem_used(min_mem_used_.begin(), min_mem_used_.end());
  for (const auto &iter : event_span_) {
    auto span = iter.second.second;
    auto &event = iter.second.first;
    auto start_index = ((event->index + total_step_ - span + 1) % total_step_);
    bool revert = false;
    size_t cur_index = start_index;
    while (cur_index != event->index) {
      cur_mem_used[cur_index] += event->mem_size;
      if (cur_mem_used[cur_index] > mem_size_) {
        revert = true;
      }
      cur_index += 1;
      if (cur_index >= total_step_) {
        cur_index = 0;
      }
    }
    if (revert) {
      cur_index = start_index;
      while (cur_index != event->index) {
        cur_mem_used[cur_index] -= event->mem_size;
        cur_index += 1;
        if (cur_index >= total_step_) {
          cur_index = 0;
        }
      }
      (void)swap_events_.emplace(event);
    }
  }
}

void MemOffloadStrategy::GenComputeMemEvents() {
  pre_compute_events_.clear();
  post_compute_events_.clear();
  pre_compute_events_.resize(total_step_);
  post_compute_events_.resize(total_step_);
  for (auto &item : mem_events_) {
    auto &mem_events = item.second;
    // No need to generate events for memory that has only one event, which means it is never used by any kernel.
    if (mem_events.size() <= 1) {
      continue;
    }

    const bool is_high_priority = IsHighPriorityMem(item.first);
    auto first_event = mem_events[kInitOrMallocMemEventIndex];
    MS_EXCEPTION_IF_NULL(first_event);
    const auto &first_get_event = mem_events[kFirstGetMemEventIndex];
    MS_EXCEPTION_IF_NULL(first_get_event);
    if (is_high_priority && swap_events_.find(first_get_event) != swap_events_.end()) {
      first_event->index = first_get_event->index;
    }
    if ((first_event->type == kInit || first_event->type == kMalloc) && first_event->index < total_step_) {
      (void)pre_compute_events_[first_event->index].emplace_back(first_event);
    } else {
      MS_LOG_EXCEPTION << "First event should be init or malloc!";
    }

    const auto &last_event = mem_events[mem_events.size() - 1];
    size_t pre_index = is_high_priority ? last_event->index : first_event->index;
    const auto &swap_out_event_index = GetSwapOutEventIndex(item.first, mem_events);
    for (size_t i = kFirstGetMemEventIndex; i < mem_events.size(); ++i) {
      auto &event = mem_events[i];
      MS_EXCEPTION_IF_NULL(event);
      if (need_swap_ && swap_events_.find(event) != swap_events_.end()) {
        MemEventType event_type = kSwapOut;
        if (is_high_priority && swap_out_event_index.count(i) == 0) {
          event_type = kFree;
        }
        auto free_or_swap_out_event = std::make_shared<MemEvent>(event_type, pre_index);
        free_or_swap_out_event->key = item.first;
        free_or_swap_out_event->mem_size = first_event->mem_size;
        (void)post_compute_events_[pre_index].emplace_back(free_or_swap_out_event);
        // avoid swap-in-event follow init-event
        if (i != kFirstGetMemEventIndex || first_event->type != kInit) {
          auto swap_in_event = std::make_shared<MemEvent>(kSwapIn, event->index);
          swap_in_event->key = item.first;
          swap_in_event->mem_size = first_event->mem_size;
          (void)pre_compute_events_[event->index].emplace_back(swap_in_event);
        }
      }
      if (event->index < pre_compute_events_.size()) {
        (void)pre_compute_events_[event->index].emplace_back(event);
      }
      pre_index = event->index;
    }
    if (!is_high_priority) {
      GenFreeEvent(last_event);
    }
  }
}

void MemOffloadStrategy::GenFreeEvent(const std::shared_ptr<MemEvent> &last_event) {
  MS_EXCEPTION_IF_NULL(last_event);
  auto free_event = std::make_shared<MemEvent>(kFree, last_event->index);
  free_event->key = last_event->key;
  if (last_event->index < post_compute_events_.size()) {
    (void)post_compute_events_[last_event->index].emplace_back(free_event);
  }
}

std::set<size_t> MemOffloadStrategy::GetSwapOutEventIndex(const void *key,
                                                          const std::vector<std::shared_ptr<MemEvent>> &mem_events) {
  const auto &update_step_iter = high_priority_updated_step_.find(key);
  if (update_step_iter == high_priority_updated_step_.end() || update_step_iter->second.empty()) {
    return std::set<size_t>();
  }
  const auto &update_steps = update_step_iter->second;
  size_t update_steps_index = 0;
  std::set<size_t> swap_out_event_index;
  size_t min_swap_index_before_update = SIZE_MAX;
  size_t max_swap_out_step = 0;
  for (size_t i = 0; i < mem_events.size(); ++i) {
    const auto &mem_event = mem_events[i];
    if (swap_events_.count(mem_event) == 0) {
      continue;
    }
    if (mem_event->index <= update_steps[update_steps_index]) {
      if (i <= min_swap_index_before_update) {
        min_swap_index_before_update = i;
      }
    } else {
      (void)swap_out_event_index.insert(i);
      max_swap_out_step = mem_event->index;
      while (update_steps_index < update_steps.size() && update_steps[update_steps_index] < mem_event->index) {
        ++update_steps_index;
      }
    }
  }
  if (max_swap_out_step <= update_steps[update_steps.size() - 1]) {
    (void)swap_out_event_index.insert(min_swap_index_before_update);
  }
  return swap_out_event_index;
}
}  // namespace device
}  // namespace mindspore
