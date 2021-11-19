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
#include <set>
#include <memory>
#include <utility>
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
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
    GenNoSwapEventSet();
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
  for (auto &item : mem_events_) {
    auto &mem_events = item.second;
    if (mem_events.empty()) {
      continue;
    }
    auto first_event = mem_events[0];
    size_t cur_index = 0;
    if (first_event != nullptr && first_event->type == kInit && mem_events.size() > 1) {
      first_event = mem_events[1];
      cur_index = 1;
    }
    auto last_event = mem_events[mem_events.size() - 1];
    for (size_t start_index = first_event->index; start_index <= last_event->index; ++start_index) {
      if (start_index < total_step_) {
        total_mem_used[start_index] += first_event->mem_size;
      } else {
        MS_LOG(ERROR) << "Error mem event index " << start_index;
      }
    }
    for (; cur_index < mem_events.size(); ++cur_index) {
      auto &event = mem_events[cur_index];
      MS_EXCEPTION_IF_NULL(event);
      if (event->index < total_step_) {
        min_mem_used_[event->index] += first_event->mem_size;
      } else {
        MS_LOG(ERROR) << "Error mem event index " << event->index;
      }
    }
  }
  min_mem_needed_ = *(std::max_element(min_mem_used_.begin(), min_mem_used_.end()));
  mem_used_without_swap_ = *(std::max_element(total_mem_used.begin(), total_mem_used.end()));
}

void MemOffloadStrategy::CheckMemSize() {
  if (mem_size_ < min_mem_needed_) {
    MS_LOG(EXCEPTION) << "Out of memory, as available mem size is " << mem_size_ << " while graph needs at least "
                      << min_mem_needed_;
  }

  if (mem_size_ < mem_used_without_swap_) {
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
    if (tensor_events.empty()) {
      continue;
    }
    auto first_event = tensor_events[0];
    size_t cur_index = 0;
    if (first_event != nullptr && first_event->type == kInit && tensor_events.size() > 1) {
      first_event = tensor_events[1];
      cur_index = 1;
    }
    size_t last_index = first_event->index;
    for (; cur_index < tensor_events.size(); ++cur_index) {
      auto &event = tensor_events[cur_index];
      MS_EXCEPTION_IF_NULL(event);
      auto span = event->index - last_index;
      if (span > 1) {
        (void)event_span_.emplace(span, event);
      }
      last_index = event->index;
    }
  }
}

void MemOffloadStrategy::GenNoSwapEventSet() {
  no_swap_events_.clear();
  std::vector<size_t> cur_mem_used(min_mem_used_.begin(), min_mem_used_.end());
  for (auto iter = event_span_.begin(); iter != event_span_.end(); ++iter) {
    auto span = iter->first;
    auto &event = iter->second;
    auto start_index = event->index - span + 1;
    bool revert = false;
    for (size_t i = start_index; i < event->index; ++i) {
      cur_mem_used[i] += event->mem_size;
      if (cur_mem_used[i] > mem_size_) {
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

void MemOffloadStrategy::GenComputeMemEvents() {
  pre_compute_events_.clear();
  post_compute_events_.clear();
  pre_compute_events_.resize(total_step_);
  post_compute_events_.resize(total_step_);
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
      pre_compute_events_[first_event->index].emplace_back(first_event);
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
        auto swap_out_event = std::make_shared<MemEvent>(kSwapOut, pre_index);
        swap_out_event->key = item.first;
        swap_out_event->mem_size = first_event->mem_size;
        post_compute_events_[pre_index].emplace_back(swap_out_event);
        auto swap_in_event = std::make_shared<MemEvent>(kSwapIn, event->index);
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
    auto free_event = std::make_shared<MemEvent>(kFree, last_event->index);
    free_event->key = item.first;
    if (last_event->index < post_compute_events_.size()) {
      (void)post_compute_events_[last_event->index].emplace_back(free_event);
    }
  }
}
}  // namespace device
}  // namespace mindspore
