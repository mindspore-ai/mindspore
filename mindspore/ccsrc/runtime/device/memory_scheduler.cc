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
#ifdef _MSC_VER
#include <time.h>
#else
#include <sys/time.h>
#endif

namespace mindspore {
namespace device {
namespace {
constexpr float kMaxMemReuseFactor = 0.9;
constexpr float kMinMemReuseFactor = 0.5;
constexpr float kRetryFactor = 0.1;

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

void MemScheduler::Record(const void *key, const MemEventType &event_type, size_t mem_size) {
  if (key == nullptr) {
    return;
  }
  auto event = std::make_shared<MemEvent>(event_type, current_step_);
  event->mem_size = mem_size;
  event->key = key;
  mem_events_[key].emplace_back(event);
  if (step_events_.size() < current_step_ + 1) {
    step_events_.resize(current_step_ + 1);
  }
  step_events_[current_step_].emplace_back(event);
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
    } else {
      Record(key, kGet, mem_size);
    }
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
  } else {
    MS_LOG_EXCEPTION << "Mem extender get nullptr result!";
  }
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
    if (event->type == kInit || event->type == kMalloc) {
      auto priority = mem_priority_[event->key];
      auto iter = high_priority_device_ptr_.find(event->key);
      if (priority != kMemPriorityLow && iter != high_priority_device_ptr_.end()) {
        MS_EXCEPTION_IF_NULL(iter->second);
        mem_result_[event->key] = iter->second;
        continue;
      }
      auto device_ptr = mem_handler_->MallocDevice(event->mem_size);
      if (device_ptr == nullptr) {
        return false;
      }
      if (priority != kMemPriorityLow) {
        high_priority_device_ptr_[event->key] = device_ptr;
      }

      if (event->type == kInit) {
        auto host_ptr = init_host_ptr_[event->key];
        MS_EXCEPTION_IF_NULL(host_ptr);
        mem_handler_->SwapIn(host_ptr, device_ptr, event->mem_size, stream);
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

  if (record_compute_time_ && !updated_) {
    compute_start_time_ = GetCurrentTime();
  }
  return true;
}

bool MemScheduler::PostCompute(void *stream) {
  if (strategy_ == nullptr) {
    ++current_step_;
    return true;
  }

  if (record_compute_time_ && !updated_) {
    compute_time_[current_step_] = GetCurrentTime() - compute_start_time_;
  }

  auto &events = strategy_->GetPostComputeEvents(current_step_);
  for (auto &event : events) {
    MS_EXCEPTION_IF_NULL(event);
    MS_LOG(DEBUG) << "Post compute " << current_step_ << ": " << event->key << " v " << event->type;
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
      (void)mem_result_.erase(event->key);
    }
  }
  ++current_step_;
  return true;
}

void MemScheduler::OptMemUsage(float mem_used_factor) {
  mem_used_factor_ = mem_used_factor;
  MS_EXCEPTION_IF_NULL(mem_handler_);

  if (strategy_ == nullptr) {
    strategy_ = std::make_shared<MemOffloadStrategy>(mem_priority_, mem_events_, total_step_);
    compute_time_.resize(total_step_);
  }

  auto available_mem_size = mem_handler_->GetAvailableMemSize();
  available_mem_size = available_mem_size * mem_used_factor_;
  strategy_->set_mem_size(available_mem_size);
  strategy_->Execute();
}

void MemScheduler::Optimize() {
  float mem_used_factor = kMaxMemReuseFactor;
  while (!optimized_ && mem_used_factor >= kMinMemReuseFactor) {
    OptMemUsage(mem_used_factor);
    current_step_ = 0;
    bool ret = true;
    for (size_t step = 0; step < total_step_; ++step) {
      ret = PreCompute(nullptr);
      auto &step_events = step_events_[step];
      for (auto &event : step_events) {
        if (event->type != kGet) {
          continue;
        }
        auto ptr = GetOrMalloc(event->key, event->mem_size);
        if (ptr == nullptr) {
          ret = false;
          break;
        }
      }
      if (!ret) {
        break;
      }
      PostCompute(nullptr);
    }
    if (ret) {
      optimized_ = true;
    } else {
      mem_used_factor -= kRetryFactor;
    }
  }
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
