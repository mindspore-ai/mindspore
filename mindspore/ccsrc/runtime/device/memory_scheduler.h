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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_SCHEDULER_H_
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <utility>

namespace mindspore {
namespace device {
class MemHandler {
 public:
  virtual size_t GetAvailableMemSize() = 0;
  virtual void *MallocDevice(size_t mem_size) = 0;
  virtual void FreeDevice(void *ptr) = 0;
  virtual void *MallocHost(size_t mem_size) = 0;
  virtual void FreeHost(void *ptr) = 0;
  virtual void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) = 0;
  virtual void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) = 0;
};

enum MemPriority { kMemPriorityLow, kMemPriorityMedium, kMemPriorityHigh };

class MemScheduler {
  enum EventType { kInit, kMalloc, kGet, kFree, kSwapIn, kSwapOut };

  struct Event {
    Event(const EventType &in_type, size_t in_index) {
      type = in_type;
      index = in_index;
    }

    EventType type;
    size_t index{0};
    size_t mem_size{0};
    const void *key{nullptr};
  };

 public:
  MemScheduler() = default;
  ~MemScheduler() = default;

  bool need_record_event() const { return need_record_event_; }

  bool optimized() const { return optimized_; }

  void SetOptimized(bool flag) { optimized_ = flag; }

  void SetMemHandler(const std::shared_ptr<MemHandler> &handler) { mem_handler_ = handler; }

  void Init(const void *key, void *host_ptr, size_t mem_size, MemPriority priority = kMemPriorityLow);

  void *GetOrMalloc(const void *key, size_t mem_size, MemPriority priority = kMemPriorityLow);

  void RecordMemUsage() { compute_index_ = 0; }

  bool PreCompute(void *stream);

  bool PostCompute(void *stream);

  void OptMemUsage();

  void Clear();

  bool IsHighPriorityMem(const void *key);

  void SetMemPriority(const void *key, MemPriority priority);

  void SetMemUsedFactor(float factor) { mem_used_factor_ = factor; }

  void SetNeedSwap(bool flag) { need_swap_ = flag; }

 private:
  void Record(const void *key, const EventType &event_type, size_t mem_size = 0);
  void GenEvents();
  void CheckMemSize();
  void CountMemUsage();
  void GenEventSpan();
  void GenNoSwapEventSet();
  std::map<const void *, MemPriority> mem_priority_;
  std::map<const void *, std::vector<std::shared_ptr<Event>>> mem_events_;
  std::vector<std::vector<std::shared_ptr<Event>>> pre_compute_events_;
  std::vector<std::vector<std::shared_ptr<Event>>> post_compute_events_;
  std::map<const void *, void *> mem_result_;
  std::map<const void *, void *> init_host_ptr_;
  std::map<const void *, void *> swap_host_ptr_;
  std::map<const void *, void *> high_priority_device_ptr_;
  size_t compute_index_{0};
  bool need_record_event_{true};
  bool optimized_{false};
  std::shared_ptr<MemHandler> mem_handler_{nullptr};
  bool need_swap_{false};
  std::multimap<size_t, std::shared_ptr<Event>> event_span_;
  std::set<std::shared_ptr<Event>> no_swap_events_;
  std::vector<size_t> min_mem_used_;
  size_t mem_used_without_swap_{0};
  size_t min_mem_needed_{0};
  float mem_used_factor_{0.9};
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
