/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include <algorithm>
#include <iterator>
#include <limits>
#include "minddata/dataset/engine/cache/cache_hw.h"
#include "minddata/dataset/engine/cache/cache_numa.h"
#include "minddata/dataset/util/random.h"
namespace mindspore {
namespace dataset {
NumaMemoryPool::NumaMemoryPool(std::shared_ptr<CacheServerHW> hw, float memory_cap_ratio)
    : hw_(std::move(hw)), memory_cap_ratio_(memory_cap_ratio) {
  int64_t total_avail = 0;
  // We will create a number of small Arenas to spread out the server threads so it
  // will be less contention. If we link with the numa library, i.e. if
  // NUMA_ENABLED is defined, we will make use of the low level numa library such that
  // each Arena solely comes from one particular socket.
  // The total number of Arenas will be controlled under the number of cpus.
  auto num_cpus = hw_->GetCpuCount();
  memory_segments_.reserve(num_cpus);
  arena_list_.reserve(num_cpus);
  mux_ = std::make_unique<std::mutex[]>(num_cpus);
  auto num_memory_nodes = num_cpus;
  int64_t max_avail = CacheServerHW::GetTotalSystemMemory() * memory_cap_ratio_;
  int64_t arena_sz = max_avail / num_memory_nodes;
  // If arena_sz is too small, lower the number of Arenas.
  if (arena_sz < std::numeric_limits<int32_t>::max()) {
    arena_sz = round_up_4K(std::numeric_limits<int32_t>::max());
    num_memory_nodes = max_avail / arena_sz;
    if (num_memory_nodes == 0) {
      num_memory_nodes = 1;
      arena_sz = max_avail;
    }
  }
  MS_LOG(INFO) << "Creating " << num_memory_nodes << " number of arena. Each one of size " << arena_sz;

#ifdef NUMA_ENABLED
  if (numa_available() != -1) {
    auto num_numa_nodes = hw_->GetNumaNodeCount();
    numa_id_t node_id = 0;
    for (auto i = 0; i < num_memory_nodes; ++i) {
      auto success = CreateMultipleArenas(arena_sz, node_id++ % num_numa_nodes, 1);
      total_avail += success * arena_sz;
    }
  } else {
    auto success = CreateMultipleArenas(arena_sz, 0, num_memory_nodes);
    total_avail += success * arena_sz;
  }
#else
  auto success = CreateMultipleArenas(arena_sz, 0, num_memory_nodes);
  total_avail += success * arena_sz;
#endif
  memory_cap_ = total_avail;
  MS_LOG(WARNING) << "Memory pool created. Total available memory " << memory_cap_ << " spread in " << nodes_.size()
                  << " arenas";
  int32_t slot = 0;
  // Set up a map for future easy access.
  for (auto node_id : nodes_) {
    numa_map_[node_id].push_back(slot);
    ++slot;
  }
}

int32_t NumaMemoryPool::CreateMultipleArenas(int64_t segment_sz, numa_id_t node_id, int32_t repeat_count) {
  int32_t success = 0;
  for (auto i = 0; i < repeat_count; ++i) {
#ifdef NUMA_ENABLED
    void *ptr = numa_alloc_onnode(segment_sz, node_id);
#else
    void *ptr = malloc(segment_sz);
#endif
    if (ptr != nullptr) {
      memory_segments_.emplace_back(ptr, segment_sz);
      arena_list_.push_back(std::make_unique<ArenaImpl>(ptr, segment_sz));
      nodes_.push_back(node_id);
      ++success;
    } else {
      // Skip the rest.
      break;
    }
  }
  MS_LOG(DEBUG) << "Allocate " << success << " arenas from node " << node_id;
  return success;
}

NumaMemoryPool::~NumaMemoryPool() {
  if (!memory_segments_.empty()) {
    for (auto &s : memory_segments_) {
#ifdef NUMA_ENABLED
      numa_free(s.first, s.second);
#else
      free(s.first);
#endif
    }
  }
}

Status NumaMemoryPool::Allocate(size_t n, void **p) {
  RETURN_UNEXPECTED_IF_NULL(p);
  auto mt = GetRandomDevice();
  Status rc;
  void *ptr = nullptr;
  auto num_segments = memory_segments_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(num_segments > 0, "No numa nodes available");
  if (NumaAware()) {
    auto num_numa_nodes = hw_->GetNumaNodeCount();
    // We will start from the numa node this worker id is running on and do a round robin search.
    numa_id_t start = hw_->GetMyNode();
    numa_id_t node_id = start;
    do {
      auto it = numa_map_.find(node_id);
      if (it != numa_map_.end()) {
        auto &slots = it->second;
        auto num_slots = slots.size();
        std::uniform_int_distribution<int32_t> distribution(0, num_slots - 1);
        auto start_slot = distribution(mt);
        int32_t inx = start_slot;
        do {
          int32_t k = slots.at(inx);
          std::unique_lock lock_x(mux_[k]);
          auto &impl = arena_list_.at(k);
          rc = impl->Allocate(n, &ptr);
          if (rc.IsOk()) {
            *p = ptr;
            break;
          } else if (rc == StatusCode::kMDOutOfMemory) {
            inx = (inx + 1) % num_slots;
          } else {
            return rc;
          }
        } while (inx != start_slot);
      }
      // We have done searching for this numa node. If not found, move to the next node.
      if (ptr == nullptr) {
        node_id = (node_id + 1) % num_numa_nodes;
      } else {
        break;
      }
    } while (node_id != start);
  } else {
    // If not numa aware, just randomly pick a slot.
    std::uniform_int_distribution<int32_t> distribution(0, num_segments - 1);
    auto start_slot = distribution(mt);
    int32_t slot = start_slot;
    do {
      std::unique_lock lock_x(mux_[slot]);
      auto &impl = arena_list_.at(slot);
      rc = impl->Allocate(n, &ptr);
      if (rc.IsOk()) {
        *p = ptr;
        break;
      } else if (rc == StatusCode::kMDOutOfMemory) {
        // Make the next arena and continue.
        slot = (slot + 1) % num_segments;
      } else {
        return rc;
      }
    } while (slot != start_slot);
  }
  // Handle the case we have done one round robin search.
  if (ptr == nullptr) {
    return Status(StatusCode::kMDOutOfMemory, __LINE__, __FILE__);
  }
  return rc;
}

void NumaMemoryPool::Deallocate(void *p) {
  // Find out which numa slot it comes from.
  auto slot = Locate(p);
  MS_ASSERT(slot != -1);
  std::unique_lock lock_x(mux_[slot]);
  auto &impl = arena_list_.at(slot);
  impl->Deallocate(p);
}

int NumaMemoryPool::PercentFree() const {
  int percent_free = 0;
  int num_arena = 0;
  for (auto const &p : arena_list_) {
    percent_free += p->PercentFree();
    num_arena++;
  }
  if (num_arena) {
    return percent_free / num_arena;
  } else {
    return 100;
  }
}

int32_t NumaMemoryPool::Locate(void *p) const {
  int32_t slot = 0;
  char *mem = reinterpret_cast<char *>(p);
  for (slot = 0; slot < memory_segments_.size(); ++slot) {
    auto elem = memory_segments_.at(slot);
    char *q = reinterpret_cast<char *>(elem.first);
    if (mem >= q && mem < q + elem.second) {
      return slot;
    }
  }
  return -1;
}

std::vector<numa_id_t> NumaMemoryPool::GetAvailableNodes() const {
  std::vector<numa_id_t> v;
  std::transform(numa_map_.begin(), numa_map_.end(), std::back_inserter(v),
                 [](const std::pair<numa_id_t, std::vector<int32_t>> &v) { return v.first; });
  return v;
}

}  // namespace dataset
}  // namespace mindspore
