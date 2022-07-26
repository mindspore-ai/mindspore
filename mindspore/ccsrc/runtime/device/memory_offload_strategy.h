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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_OFFLOAD_STRATEGY_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_OFFLOAD_STRATEGY_H_
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <utility>
#include <algorithm>

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

using MemEventPtr = std::shared_ptr<MemEvent>;
using MemEventPtrList = std::vector<MemEventPtr>;

struct ContinuousMemInfo {
  ContinuousMemInfo(bool is_input, size_t total_size, size_t compute_index, std::vector<size_t> align_size_list)
      : is_input_(is_input),
        total_size_(total_size),
        compute_index_(compute_index),
        align_size_list_(std::move(align_size_list)) {}
  bool is_input_;
  size_t total_size_;
  size_t compute_index_;
  const std::vector<size_t> align_size_list_;
  std::map<const void *, size_t> key_index_map_;
};

using ContinuousMemInfoPtr = std::shared_ptr<ContinuousMemInfo>;

class ContinuousMemInfoHelper {
 public:
  void AddContinuousMemInfo(bool is_input, size_t compute_index, size_t total_size,
                            const std::vector<size_t> &align_size_list,
                            const std::vector<const void *> &address_key_list);
  std::shared_ptr<ContinuousMemInfo> GetContinuousMemInfo(const void *address_key) const;
  std::vector<ContinuousMemInfoPtr> GetAllContinuousMemInfo() const;
  bool IsContinuousMem(const void *address_key) const;
  bool IsContinuousInputMem(const void *address_key) const;

  void AddContinuousMallocIndex(const ContinuousMemInfoPtr &mem_info, size_t index) {
    (void)first_malloc_index_.emplace(mem_info, index);
  }

  bool NeedMallocContinuousMem(const ContinuousMemInfoPtr &mem_info, size_t index) const {
    const auto &iter = first_malloc_index_.find(mem_info);
    return iter != first_malloc_index_.end() && iter->second == index;
  }

  void ClearContinuousMallocIndex() { first_malloc_index_.clear(); }

  const std::vector<ContinuousMemInfoPtr> &GetIndexContinuousMemInfo(size_t step) {
    return index_continuous_info_map_[step];
  }

 private:
  std::set<ContinuousMemInfoPtr> input_continuous_mem_info_;
  std::set<ContinuousMemInfoPtr> output_continuous_mem_info_;
  std::map<const void *, ContinuousMemInfoPtr> key_continuous_info_map_;
  std::map<ContinuousMemInfoPtr, size_t> first_malloc_index_;
  std::map<size_t, std::vector<ContinuousMemInfoPtr>> index_continuous_info_map_;
};

class MemOffloadStrategy {
 public:
  MemOffloadStrategy(const std::map<const void *, MemPriority> &mem_priority,
                     const std::map<const void *, MemEventPtrList> &mem_events,
                     const std::set<const void *> &manual_offload_keys, size_t total_step,
                     std::shared_ptr<ContinuousMemInfoHelper> continuous_mem_info_manager)
      : mem_priority_(mem_priority),
        mem_events_(mem_events),
        manual_offload_keys_(manual_offload_keys),
        total_step_(total_step),
        continuous_mem_info_helper_(std::move(continuous_mem_info_manager)) {}

  virtual ~MemOffloadStrategy() = default;

  virtual void Execute();

  void SetComputeTime(const std::vector<double> &compute_time) { compute_time_ = compute_time; }

  MemEventPtrList &GetPreComputeEvents(size_t step);

  MemEventPtrList &GetPostComputeEvents(size_t step);

  void set_mem_size(size_t mem_size) { mem_size_ = mem_size; }

  bool need_swap() const { return need_swap_; }

 private:
  bool IsHighPriorityMem(const void *key) const;

  void CountMemUsage();

  void CheckMemSize();

  void GenEventSpan();

  void GenSwapEventSet();

  void GenComputeMemEvents();

  void GenFreeEvent(const MemEventPtr &last_event);

  void AddToSwapEventSetIfOutOfMem(const MemEventPtr &mem_event, size_t span, std::vector<size_t> *mem_used);

  void GenContinuousMemSwapEvent(const ContinuousMemInfoPtr &continuous_mem_info, std::vector<size_t> *mem_used,
                                 std::set<MemEventPtr> *events_no_need_swap);

  size_t GetMaxSpanForContinuousMem(const ContinuousMemInfoPtr &continuous_mem_info,
                                    const std::vector<size_t> &mem_used) const;

  size_t GetFirstMallocIndex(const ContinuousMemInfoPtr &continuous_mem_info) const;

  void GenContinuousMemAllocSteps();

  void GenContinuousMemAllocStep(const ContinuousMemInfoPtr &continuous_mem_info);

  void CountContinuousMemUsage(std::vector<size_t> *total_mem_used) const;

  size_t GetSpanBetweenMemEvents(size_t pre_index, size_t post_index) const {
    return (post_index + total_step_ - pre_index) % total_step_;
  }

  size_t GetPreMemEventIndex(size_t cur_index, size_t span) const {
    return (cur_index + total_step_ - span) % total_step_;
  }

  const std::map<const void *, MemPriority> &mem_priority_;
  const std::map<const void *, MemEventPtrList> &mem_events_;
  const std::set<const void *> &manual_offload_keys_;
  const size_t total_step_;
  std::vector<MemEventPtrList> pre_compute_events_;
  std::vector<MemEventPtrList> post_compute_events_;

  size_t mem_size_{0};
  std::vector<double> compute_time_;
  bool need_swap_{false};
  std::multimap<size_t, std::pair<MemEventPtr, size_t>> event_span_;
  std::multimap<size_t, std::pair<MemEventPtr, size_t>> continuous_input_event_span_;
  std::set<MemEventPtr> swap_events_;
  std::vector<size_t> min_mem_used_;
  size_t mem_used_without_swap_{0};
  size_t min_mem_needed_{0};
  std::shared_ptr<ContinuousMemInfoHelper> continuous_mem_info_helper_{nullptr};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_OFFLOAD_STRATEGY_H_
