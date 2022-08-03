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
#include <queue>
#include <utility>
#include "runtime/device/memory_offload_strategy.h"
#include "runtime/device/memory_manager.h"

namespace mindspore {
namespace device {
class MemHandler {
 public:
  explicit MemHandler(const std::shared_ptr<MemoryManager> &memory_manager) : memory_manager_(memory_manager) {}
  ~MemHandler() = default;
  size_t GetAvailableMemSize() { return memory_manager_->GetAvailableMemSize(); }
  void *MallocDevice(size_t mem_size) { return memory_manager_->MallocMemFromMemPool(mem_size, false); }
  void FreeDevice(void *ptr) { memory_manager_->FreeMemFromMemPool(ptr); }
  void *MallocHost(size_t mem_size);
  void FreeHost(void *ptr);
  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
    memory_manager_->SwapIn(host_ptr, device_ptr, mem_size, stream);
  }
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
    memory_manager_->SwapOut(device_ptr, host_ptr, mem_size, stream);
  }
  std::vector<void *> MallocContinuousMemFromMemPool(const std::vector<size_t> &size_list) {
    return memory_manager_->MallocContinuousMemFromMemPool(size_list);
  }

 private:
  std::shared_ptr<MemoryManager> memory_manager_;
  std::map<size_t, std::queue<void *>> cached_host_mem_;
  std::map<void *, std::shared_ptr<std::vector<uint8_t>>> host_mem_block_map_;
};

class MemScheduler {
 public:
  MemScheduler() = default;
  ~MemScheduler() = default;

  bool need_record_event() const { return need_record_event_; }

  void set_need_record_event(bool flag) { need_record_event_ = flag; }

  bool optimized() const { return optimized_; }

  void Update();

  void SetMemHandler(const std::shared_ptr<MemHandler> &handler) { mem_handler_ = handler; }

  void Init(const void *key, void *host_ptr, size_t mem_size, MemPriority priority = kMemPriorityLow);

  void *GetOrMalloc(const void *key, size_t mem_size, MemPriority priority = kMemPriorityLow);

  bool HasDeviceMem(const void *key) const { return mem_result_.find(key) != mem_result_.end(); }

  void UpdateHighPriorityMem(const void *key) { (void)updated_high_priority_mem_.insert(key); }

  void SetTotalStep(size_t step) {
    total_step_ = step;
    step_keys_.resize(total_step_);
  }

  void Reset() { current_step_ = 0; }

  bool PreCompute(void *stream);

  bool PostCompute(void *stream);

  bool Optimize();

  void Clear();

  void ClearAllocatedMem();

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

  void AdjustFirstEventIndex();

  void *MallocDevice(size_t mem_size, void *stream);

  std::vector<void *> MallocContinuousMem(size_t total_size, const std::vector<size_t> &size_list, void *stream);

  void SwapOutAndFreeDevice(const void *key, void *device_ptr, size_t mem_size, void *stream);

  size_t GetMemSize(const void *key);

  void GetOrMallocHostPtr(const void *key, size_t mem_size, void **host_ptr, bool *from_init);

  void GetHostPtr(const void *key, void **host_ptr, bool *from_init);

  bool PreComputeMock(const MemEventPtr &event);

  bool PreComputeInit(const MemEventPtr &event, void *stream);

  bool PreComputeMalloc(const MemEventPtr &event, void *stream);

  bool PreComputeSwapIn(const MemEventPtr &event, void *stream);

  bool PreComputeGet(const MemEventPtr &event, void *stream);

  void *MallocContinuousMem(const MemEventPtr &event, void *stream);

  // Scheduler status
  bool need_record_event_{true};
  bool optimized_{false};
  bool updated_{false};
  bool record_compute_time_{false};
  // Memory status
  std::map<const void *, void *> mem_result_;
  std::map<const void *, MemPriority> mem_priority_;
  std::map<const void *, MemEventPtrList> mem_events_;
  std::vector<std::set<const void *>> step_keys_;
  std::set<const void *> high_priority_mem_need_init_;
  std::set<const void *> updated_high_priority_mem_;
  std::shared_ptr<ContinuousMemInfoHelper> continuous_mem_info_helper_{std::make_shared<ContinuousMemInfoHelper>()};
  std::set<std::shared_ptr<ContinuousMemInfo>> cur_step_allocated_continuous_mem_;
  std::set<const void *> continuous_mem_key_;
  size_t total_step_{0};
  size_t current_step_{0};
  std::set<const void *> manual_offload_keys_;
  std::map<const void *, void *> init_host_ptr_;
  std::map<const void *, void *> swap_host_ptr_;
  // Compute time
  std::vector<double> compute_time_;
  double compute_start_time_{0};

  std::shared_ptr<MemHandler> mem_handler_{nullptr};
  std::shared_ptr<MemOffloadStrategy> strategy_{nullptr};
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
