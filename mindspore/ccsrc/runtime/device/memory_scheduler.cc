/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
  auto event = std::make_shared<MemEvent<const void *>>(event_type, current_step_);
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
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  auto_mem_offload_->SetInitHostPtr(key, host_ptr, mem_size);
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
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  return auto_mem_offload_->Get(key);
}

void *MemScheduler::Malloc(const MemEventPtr<const void *> &event, void *stream) {
  MS_EXCEPTION_IF_NULL(event);
  MS_EXCEPTION_IF_NULL(continuous_mem_info_helper_);
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  const bool is_continuous_mem = continuous_mem_info_helper_->IsContinuousMem(event->key);
  if (!is_continuous_mem) {
    return auto_mem_offload_->Malloc(event->key, event->mem_size, stream, GetNoReuseKeys());
  }
  const auto &continuous_mem_info = continuous_mem_info_helper_->GetContinuousMemInfo(event->key);
  MS_EXCEPTION_IF_NULL(continuous_mem_info);
  if (!continuous_mem_info_helper_->NeedMallocContinuousMem(continuous_mem_info, current_step_) ||
      cur_step_allocated_continuous_mem_.count(continuous_mem_info) != 0) {
    return auto_mem_offload_->Malloc(event->key, event->mem_size, stream, GetNoReuseKeys());
  }
  std::vector<const void *> keys(continuous_mem_info->key_index_map_.size(), nullptr);
  for (const auto &iter : continuous_mem_info->key_index_map_) {
    if (auto_mem_offload_->Get(iter.first, stream, GetNoReuseKeys()) != nullptr) {
      MS_LOG(EXCEPTION) << "Device memory is allocated before first continuous memory alloc event, event key: "
                        << event->key << ", continuous memory used index: " << continuous_mem_info->compute_index_;
    }
    keys[iter.second] = iter.first;
  }
  if (!auto_mem_offload_->MallocContinuous(keys, continuous_mem_info->align_size_list_, stream, GetNoReuseKeys())) {
    MS_LOG(WARNING) << "Alloc continuous memory failed, size: " << continuous_mem_info->total_size_;
    return nullptr;
  }
  (void)cur_step_allocated_continuous_mem_.insert(continuous_mem_info);
  return auto_mem_offload_->Get(event->key);
}

bool MemScheduler::PreComputeMock(const MemEventPtr<const void *> &event) {
  MS_EXCEPTION_IF_NULL(event);
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  void *device_ptr = nullptr;
  if (auto_mem_offload_->Get(event->key) != nullptr) {
    return true;
  } else {
    device_ptr = Malloc(event, nullptr);
  }
  return device_ptr != nullptr;
}

bool MemScheduler::PreComputeInit(const MemEventPtr<const void *> &event, void *stream) {
  MS_EXCEPTION_IF_NULL(event);
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  auto device_ptr = auto_mem_offload_->Get(event->key);
  const bool new_malloc = device_ptr == nullptr;
  if (new_malloc) {
    device_ptr = Malloc(event, stream);
  }
  if (device_ptr == nullptr) {
    return false;
  }
  if (new_malloc || high_priority_mem_need_init_.count(event->key) != 0) {
    MS_LOG(DEBUG) << "Init input data from host, key: " << event->key;
    (void)auto_mem_offload_->SwapIn(event->key, stream);
  }
  return true;
}

bool MemScheduler::PreComputeMalloc(const MemEventPtr<const void *> &event, void *stream) {
  return Malloc(event, stream) != nullptr;
}

bool MemScheduler::PreComputeSwapIn(const MemEventPtr<const void *> &event, void *stream) {
  MS_EXCEPTION_IF_NULL(event);
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  if (Malloc(event, stream) == nullptr) {
    return false;
  }
  return auto_mem_offload_->SwapIn(event->key, stream) != nullptr;
}

bool MemScheduler::PreComputeGet(const MemEventPtr<const void *> &event, void *stream) {
  MS_EXCEPTION_IF_NULL(event);
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  return auto_mem_offload_->Get(event->key, stream, GetNoReuseKeys()) != nullptr;
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
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  for (auto &event : events) {
    MS_EXCEPTION_IF_NULL(event);
    MS_LOG(DEBUG) << "Post compute " << current_step_ << ": " << event->key << " v " << event->type;
    if (event->type == kSwapOut && optimized_) {
      auto_mem_offload_->SwapOut(event->key, stream);
    }
    auto_mem_offload_->Free(event->key);
  }
  ++current_step_;
  return true;
}

void MemScheduler::OptMemUsage(float mem_used_factor) {
  MS_EXCEPTION_IF_NULL(mem_handler_);
  MS_EXCEPTION_IF_NULL(auto_mem_offload_);
  if (strategy_ == nullptr) {
    strategy_ = std::make_shared<MemOffloadStrategy<const void *>>(mem_priority_, mem_events_, manual_offload_keys_,
                                                                   total_step_, continuous_mem_info_helper_);
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
    Clear();
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
