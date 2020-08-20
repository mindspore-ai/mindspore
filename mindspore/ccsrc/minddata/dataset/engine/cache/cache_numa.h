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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_NUMA_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_NUMA_H_

#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/cache/cache_hw.h"
#include "minddata/dataset/util/arena.h"
#include "minddata/dataset/util/memory_pool.h"

namespace mindspore {
namespace dataset {
/// \brief An allocator but for a particular numa node.
template <typename T>
class NumaAllocator {
 public:
  explicit NumaAllocator(numa_id_t node_id, CachePoolPolicy policy)
      : policy_(policy), numa_enabled_(false), node_id_(node_id) {
#ifdef NUMA_ENABLED
    numa_enabled_ = numa_available() != -1;
#endif
  }
  ~NumaAllocator() = default;

  template <typename U>
  explicit NumaAllocator(NumaAllocator<U> const &rhs)
      : policy_(rhs.policy_), numa_enabled_(rhs.numa_enabled_), node_id_(rhs.node_id_) {}

  template <typename U>
  bool operator==(Allocator<U> const &rhs) const {
    return node_id_ == rhs.node_id_;
  }

  template <typename U>
  bool operator!=(Allocator<U> const &rhs) const {
    return node_id_ != rhs.node_id_;
  }

  template <typename U>
  friend class NumaAllocator;

  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = uint64_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    using other = Allocator<U>;
  };

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  /// Allocate memory on this node only. Return nullptr if no memory on this numa node.
  /// \note. This version will not throw if we can't allocate memory from this node.
  /// User must check if the pointer returned is null or not.
  pointer allocate(std::size_t n) noexcept {
    auto sz = n * sizeof(T);
    void *p = nullptr;
#ifdef NUMA_ENABLED
    if (numa_enabled_) {
      switch (policy_) {
        case kPreferred:
          numa_set_preferred(node_id_);
          p = numa_alloc(sz);
          break;
        case kLocal:
          p = numa_alloc_local(sz);
          break;
        case kInterleave:
          p = numa_alloc_interleaved(sz);
          break;
        case kOnNode:
          p = numa_alloc_onnode(sz, node_id_);
          break;
        case kNone:
        default:
          p = numa_alloc(sz);
          break;
      }
    } else {
      p = malloc(sz);
    }
#else
    p = malloc(sz);
#endif
    return reinterpret_cast<pointer>(p);
  }

  /// Free a memory allocated on this node.
  void deallocate(pointer p, std::size_t n) noexcept {
#ifdef NUMA_ENABLED
    if (numa_enabled_) {
      numa_free(p, n * sizeof(T));
    } else {
      free(p);
    }
#else
    free(p);
#endif
  }

  /// \brief Allow one to change to another numa node
  void SetNodeId(numa_id_t node_id) { node_id_ = node_id; }

  /// \brif Getter for node_id;
  numa_id_t GetNodeId() const { return node_id_; }

  /// \brief Getter for policy
  CachePoolPolicy GetPolicy() const { return policy_; }

 private:
  CachePoolPolicy policy_;
  bool numa_enabled_;
  numa_id_t node_id_;
};

/// \brief A NumaMemoryPool is like a CircularPool but all the arenas have already been allocated
/// and each one comes from a numa socket. Memory is allocated using OnNode policy. That is,
/// it is solely comes from one particular numa node, and is not interleaved.
class NumaMemoryPool : public MemoryPool {
 public:
  explicit NumaMemoryPool(std::shared_ptr<CacheServerHW> hw, float memory_cap_ratio);
  ~NumaMemoryPool() override;

  // As a derived class, we override the following functions
  Status Allocate(size_t size, void **pVoid) override;
  void Deallocate(void *pVoid) override;
  Status Reallocate(void **pVoid, size_t old_sz, size_t new_sz) override { RETURN_STATUS_UNEXPECTED("Not supported"); }
  uint64_t get_max_size() const override { return std::numeric_limits<uint64_t>::max(); }
  int PercentFree() const override;

  /// \brief Return if the memory pool is numa aware
  bool NumaAware() const { return CacheServerHW::numa_enabled(); }

  /// \brief. This returns all the numa nodes that we are able to allocate memory from.
  std::vector<numa_id_t> GetAvailableNodes() const;

  /// \brief. Given a pointer (allocated from this pool), return the numa node where it is located.
  /// \note. -1 is returned if not found.
  numa_id_t FindNode(void *p) const {
    auto slot = Locate(p);
    if (slot != -1) {
      return nodes_.at(slot);
    } else {
      return -1;
    }
  }

  /// \brief Return maximum available memory
  int64_t GetAvailableMemory() const { return memory_cap_; }

 private:
  std::shared_ptr<CacheServerHW> hw_;
  float memory_cap_ratio_;
  int64_t memory_cap_;
  std::vector<std::pair<void *, int64_t>> memory_segments_;
  std::vector<std::unique_ptr<ArenaImpl>> arena_list_;
  std::unique_ptr<std::mutex[]> mux_;
  std::vector<numa_id_t> nodes_;
  std::map<numa_id_t, std::vector<int32_t>> numa_map_;

  /// \brief. Returns the slot that a given memory comes from.
  /// \return slot from numa_segments. -1 if not found.
  int32_t Locate(void *p) const;

  /// If numa library is not linked, or numa_availble() return -1, we will fall back to this method.
  int32_t CreateMultipleArenas(int64_t segment_sz, numa_id_t node_id, int32_t repeat_count);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_NUMA_H_
