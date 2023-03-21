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
#include "utils/hash_map.h"
#include "utils/hash_set.h"

namespace mindspore {
namespace device {
enum MemPriority { kMemPriorityLow, kMemPriorityHigh };

enum MemEventType { kInit, kMalloc, kGet, kFree, kSwapIn, kSwapOut };

template <typename Key>
struct MemEvent {
  MemEvent(const MemEventType &in_type, size_t in_index) : type(in_type), index(in_index) {}

  MemEventType type;
  size_t index{0};
  size_t mem_size{0};
  Key key{nullptr};
};

template <typename Key>
using MemEventPtr = std::shared_ptr<MemEvent<Key>>;
template <typename Key>
using MemEventPtrList = std::vector<MemEventPtr<Key>>;

template <typename Key>
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
  std::map<Key, size_t> key_index_map_;
};

template <typename Key>
using ContinuousMemInfoPtr = std::shared_ptr<ContinuousMemInfo<Key>>;

template <typename Key>
class ContinuousMemInfoHelper {
 public:
  ContinuousMemInfoHelper() = default;
  ~ContinuousMemInfoHelper() = default;
  void AddContinuousMemInfo(bool is_input, size_t compute_index, size_t total_size,
                            const std::vector<size_t> &align_size_list, const std::vector<Key> &address_key_list);
  ContinuousMemInfoPtr<Key> GetContinuousMemInfo(Key address_key) const;
  std::vector<ContinuousMemInfoPtr<Key>> GetAllContinuousMemInfo() const;
  bool IsContinuousMem(Key address_key) const;
  bool IsContinuousInputMem(Key address_key) const;

  void AddContinuousMallocIndex(const ContinuousMemInfoPtr<Key> &mem_info, size_t index) {
    (void)first_malloc_index_.emplace(mem_info, index);
    (void)continuous_mem_first_alloc_info_[index].emplace_back(mem_info);
  }

  bool NeedMallocContinuousMem(const ContinuousMemInfoPtr<Key> &mem_info, size_t index) const {
    const auto &iter = first_malloc_index_.find(mem_info);
    return iter != first_malloc_index_.end() && iter->second == index;
  }

  std::vector<ContinuousMemInfoPtr<Key>> GetContinuousMemAllocInfo(size_t index) {
    const auto &iter = continuous_mem_first_alloc_info_.find(index);
    if (iter == continuous_mem_first_alloc_info_.end()) {
      return {};
    }
    return iter->second;
  }

  void ClearContinuousMallocIndex() { first_malloc_index_.clear(); }

  const std::vector<ContinuousMemInfoPtr<Key>> &GetIndexContinuousMemInfo(size_t index) {
    return index_continuous_info_map_[index];
  }

 private:
  std::set<ContinuousMemInfoPtr<Key>> input_continuous_mem_info_;
  std::set<ContinuousMemInfoPtr<Key>> output_continuous_mem_info_;
  std::map<Key, ContinuousMemInfoPtr<Key>> key_continuous_info_map_;
  std::map<ContinuousMemInfoPtr<Key>, size_t> first_malloc_index_;
  std::map<size_t, std::vector<ContinuousMemInfoPtr<Key>>> continuous_mem_first_alloc_info_;
  std::map<size_t, std::vector<ContinuousMemInfoPtr<Key>>> index_continuous_info_map_;
};

class MemoryOffloadConflict {
 public:
  void AddMemoryOffloadConflict(const HashSet<const void *> &conflict_set);
  const HashSet<const void *> &GetConflictMap(const void *key);
  static MemoryOffloadConflict &GetInstance();
  bool CanBeOffloaded(const void *key) { return offload_backlog_.count(key) != 0; }
  void AddOffloadBacklog(const void *key) { (void)offload_backlog_.insert(key); }

 private:
  MemoryOffloadConflict() = default;
  ~MemoryOffloadConflict() = default;
  HashSet<const void *> offload_backlog_;
  HashMap<const void *, HashSet<const void *>> conflict_map_;
};

template <typename Key>
struct GraphMemStatistic {
 public:
  GraphMemStatistic() { continuous_mem_info_helper_ = std::make_shared<device::ContinuousMemInfoHelper<Key>>(); }
  ~GraphMemStatistic() = default;
  void Record(Key key, const MemEventType &event_type, size_t mem_size, MemPriority priority, size_t index);

  std::map<Key, MemPriority> mem_priority_;
  std::map<Key, MemEventPtrList<Key>> mem_events_;
  std::set<Key> manual_offload_keys_;
  std::shared_ptr<ContinuousMemInfoHelper<Key>> continuous_mem_info_helper_;
  size_t total_compute_index_{};
};

template <typename Key>
class MemOffloadStrategy {
 public:
  MemOffloadStrategy(const std::map<Key, MemPriority> &mem_priority,
                     const std::map<Key, MemEventPtrList<Key>> &mem_events, const std::set<Key> &manual_offload_keys,
                     size_t total_compute_index,
                     std::shared_ptr<ContinuousMemInfoHelper<Key>> continuous_mem_info_manager)
      : mem_priority_(mem_priority),
        mem_events_(mem_events),
        manual_offload_keys_(manual_offload_keys),
        total_compute_index_(total_compute_index),
        continuous_mem_info_helper_(std::move(continuous_mem_info_manager)) {
    AdjustFirstEventIndex();
  }

  explicit MemOffloadStrategy(const GraphMemStatistic<Key> &mem_statistic)
      : mem_priority_(mem_statistic.mem_priority_),
        mem_events_(mem_statistic.mem_events_),
        manual_offload_keys_(mem_statistic.manual_offload_keys_),
        total_compute_index_(mem_statistic.total_compute_index_),
        continuous_mem_info_helper_(mem_statistic.continuous_mem_info_helper_) {
    AdjustFirstEventIndex();
  }

  virtual ~MemOffloadStrategy() = default;

  virtual void Execute();

  void SetComputeTime(const std::vector<double> &compute_time) { compute_time_ = compute_time; }

  MemEventPtrList<Key> &GetPreComputeEvents(size_t index);

  MemEventPtrList<Key> &GetPostComputeEvents(size_t index);

  void set_mem_size(size_t mem_size) { mem_size_ = mem_size; }

  bool need_swap() const { return need_swap_; }

  std::vector<ContinuousMemInfoPtr<Key>> GetContinuousMemAllocInfo(size_t index) {
    return continuous_mem_info_helper_->GetContinuousMemAllocInfo(index);
  }

 private:
  void AdjustFirstEventIndex();

  bool IsHighPriorityMem(Key key) const;

  void CountMemUsage();

  void CheckMemSize();

  void GenEventSpan();

  void GenSwapEventSet();

  void GenComputeMemEvents();

  void GenFreeEvent(const MemEventPtr<Key> &last_event);

  void AddToSwapEventSetIfOutOfMem(const MemEventPtr<Key> &mem_event, size_t span, std::vector<size_t> *mem_used);

  void GenContinuousMemSwapEvent(const ContinuousMemInfoPtr<Key> &continuous_mem_info, std::vector<size_t> *mem_used,
                                 std::set<MemEventPtr<Key>> *events_no_need_swap);

  size_t GetMaxSpanForContinuousMem(const ContinuousMemInfoPtr<Key> &continuous_mem_info,
                                    const std::vector<size_t> &mem_used) const;

  size_t GetFirstMallocIndex(const ContinuousMemInfoPtr<Key> &continuous_mem_info) const;

  void GenContinuousMemAllocInfo();

  void GenContinuousMemAllocInfo(const ContinuousMemInfoPtr<Key> &continuous_mem_info);

  void CountContinuousMemUsage(std::vector<size_t> *total_mem_used) const;

  size_t GetSpanBetweenMemEvents(size_t pre_index, size_t post_index) const {
    return (post_index + total_compute_index_ - pre_index) % total_compute_index_;
  }

  size_t GetPreMemEventIndex(size_t cur_index, size_t span) const {
    return (cur_index + total_compute_index_ - span) % total_compute_index_;
  }

  const std::map<Key, MemPriority> &mem_priority_;
  const std::map<Key, MemEventPtrList<Key>> &mem_events_;
  const std::set<Key> &manual_offload_keys_;
  const size_t total_compute_index_;
  std::vector<MemEventPtrList<Key>> pre_compute_events_;
  std::vector<MemEventPtrList<Key>> post_compute_events_;

  size_t mem_size_{0};
  std::vector<double> compute_time_;
  bool need_swap_{false};
  std::multimap<size_t, std::pair<MemEventPtr<Key>, size_t>> event_span_;
  std::set<MemEventPtr<Key>> swap_events_;
  std::vector<size_t> min_mem_used_;
  size_t mem_used_without_swap_{0};
  size_t min_mem_needed_{0};
  std::shared_ptr<ContinuousMemInfoHelper<Key>> continuous_mem_info_helper_{nullptr};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_OFFLOAD_STRATEGY_H_
