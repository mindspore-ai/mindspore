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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_OFFLOAD_STRATEGY_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_OFFLOAD_STRATEGY_H_
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <utility>

namespace mindspore {
namespace device {
enum MemPriority { kMemPriorityLow, kMemPriorityHigh };

enum MemEventType { kInit, kMalloc, kGet, kFree, kSwapIn, kSwapOut };

struct MemEvent {
  MemEvent(const MemEventType &in_type, size_t in_index) : type(in_type), index(in_index) {}

  MemEventType type;
  size_t index{0};
  size_t mem_size{0};
  const void *key{nullptr};
};

class MemOffloadStrategy {
 public:
  MemOffloadStrategy(const std::map<const void *, MemPriority> &mem_priority,
                     const std::map<const void *, std::vector<std::shared_ptr<MemEvent>>> &mem_events,
                     size_t total_step)
      : mem_priority_(mem_priority), mem_events_(mem_events), total_step_(total_step) {}

  virtual ~MemOffloadStrategy() = default;

  virtual void Execute();

  void SetComputeTime(const std::vector<double> &compute_time) { compute_time_ = compute_time; }

  std::vector<std::shared_ptr<MemEvent>> &GetPreComputeEvents(size_t step);

  std::vector<std::shared_ptr<MemEvent>> &GetPostComputeEvents(size_t step);

  void set_mem_size(size_t mem_size) { mem_size_ = mem_size; }

  bool need_swap() const { return need_swap_; }

  bool IsHighPriorityMem(const void *key);

 private:
  void CountMemUsage();
  void CheckMemSize();
  void GenEventSpan();
  void GenSwapEventSet();
  void GenComputeMemEvents();
  void GenFreeEvent(const std::shared_ptr<MemEvent> &last_event);

  const std::map<const void *, MemPriority> &mem_priority_;
  const std::map<const void *, std::vector<std::shared_ptr<MemEvent>>> &mem_events_;
  const size_t total_step_;
  std::vector<std::vector<std::shared_ptr<MemEvent>>> pre_compute_events_;
  std::vector<std::vector<std::shared_ptr<MemEvent>>> post_compute_events_;

  size_t mem_size_{0};
  std::vector<double> compute_time_;
  bool need_swap_{false};
  std::multimap<size_t, std::pair<std::shared_ptr<MemEvent>, size_t>> event_span_;
  std::set<std::shared_ptr<MemEvent>> swap_events_;
  std::vector<size_t> min_mem_used_;
  size_t mem_used_without_swap_{0};
  size_t min_mem_needed_{0};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_OFFLOAD_STRATEGY_H_
