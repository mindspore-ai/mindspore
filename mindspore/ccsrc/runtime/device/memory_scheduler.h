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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_SCHEDULER_H_
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <queue>
#include <utility>
#include "runtime/device/memory_offload_strategy.h"
#include "runtime/device/auto_mem_offload.h"

namespace mindspore {
namespace device {
class MemScheduler {
 public:
  MemScheduler() = default;
  ~MemScheduler() = default;

  bool need_record_event() const { return need_record_event_; }

  void set_need_record_event(bool flag) { need_record_event_ = flag; }

  bool optimized() const { return optimized_; }

  void Update();

  void SetMemHandler(const std::shared_ptr<MemHandler> &handler) {
    mem_handler_ = handler;
    auto_mem_offload_ = std::make_shared<AutoMemoryOffload>(handler);
  }

  void Init(const void *key, void *host_ptr, size_t mem_size, MemPriority priority = kMemPriorityLow);

  void *GetOrMalloc(const void *key, size_t mem_size, MemPriority priority = kMemPriorityLow);

  bool HasDeviceMem(const void *key) const { return auto_mem_offload_->Get(key) != nullptr; }

  void UpdateHighPriorityMem(const void *key) { auto_mem_offload_->UpdateHighPriorityMem(key); }

  void SetTotalStep(size_t step) {
    total_step_ = step;
    step_keys_.resize(total_step_);
  }

  void Reset() { current_step_ = 0; }

  bool PreCompute(void *stream);

  bool PostCompute(void *stream);

  bool Optimize();

  void Clear() { auto_mem_offload_->Clear(); }

  void SetOffload(const void *key) { (void)manual_offload_keys_.insert(key); }

  void AddMemNeedInit(const void *key) { (void)high_priority_mem_need_init_.insert(key); }

  void ClearMemNeedInit() { high_priority_mem_need_init_.clear(); }

  void AddContinuousMemInfo(bool is_input, size_t compute_index, size_t total_size,
                            const std::vector<size_t> &align_size_list,
                            const std::vector<const void *> &address_key_list);

 private:
  void Record(const void *key, const MemEventType &event_type, size_t mem_size = 0);

  void OptMemUsage(float mem_used_factor = 1.0f);

  bool Mock();

  bool PreComputeMock(const MemEventPtr<const void *> &event);

  bool PreComputeInit(const MemEventPtr<const void *> &event, void *stream);

  bool PreComputeMalloc(const MemEventPtr<const void *> &event, void *stream);

  bool PreComputeSwapIn(const MemEventPtr<const void *> &event, void *stream);

  bool PreComputeGet(const MemEventPtr<const void *> &event, void *stream);

  const HashSet<const void *> &GetNoReuseKeys() const { return step_keys_[current_step_]; }

  void *Malloc(const MemEventPtr<const void *> &event, void *stream);

  // Scheduler status
  bool need_record_event_{true};
  bool optimized_{false};
  bool updated_{false};
  bool record_compute_time_{false};
  size_t total_step_{0};
  size_t current_step_{0};
  // Memory status
  std::map<const void *, MemEventPtrList<const void *>> mem_events_;
  std::map<const void *, MemPriority> mem_priority_;
  std::vector<HashSet<const void *>> step_keys_;
  std::set<const void *> high_priority_mem_need_init_;
  std::shared_ptr<ContinuousMemInfoHelper<const void *>> continuous_mem_info_helper_{
    std::make_shared<ContinuousMemInfoHelper<const void *>>()};
  std::set<ContinuousMemInfoPtr<const void *>> cur_step_allocated_continuous_mem_;
  std::set<const void *> manual_offload_keys_;
  // Compute time
  std::vector<double> compute_time_;
  double compute_start_time_{0};

  std::shared_ptr<AutoMemoryOffload> auto_mem_offload_;
  std::shared_ptr<MemHandler> mem_handler_{nullptr};
  std::shared_ptr<MemOffloadStrategy<const void *>> strategy_{nullptr};
};

class MemSchedulerManager {
 public:
  MemSchedulerManager() = default;
  ~MemSchedulerManager() = default;
  std::shared_ptr<MemScheduler> GetOrCreateMemScheduler(uint64_t uid) {
    auto scheduler = GetMemScheduler(uid);
    if (scheduler == nullptr) {
      scheduler = std::make_shared<MemScheduler>();
      graph_mem_scheduler_map_[uid] = scheduler;
    }
    return scheduler;
  }

  std::shared_ptr<MemScheduler> GetMemScheduler(uint64_t uid) {
    auto iter = graph_mem_scheduler_map_.find(uid);
    if (iter != graph_mem_scheduler_map_.end()) {
      return iter->second;
    }
    return nullptr;
  }

 private:
  std::map<uint64_t, std::shared_ptr<MemScheduler>> graph_mem_scheduler_map_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_SCHEDULER_H_
