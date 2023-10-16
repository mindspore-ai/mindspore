/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_HASH_MAP_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_HASH_MAP_H_

#include <cmath>
#include <utility>
#include <memory>
#include <vector>
#include <list>
#include <mutex>
#include "utils/hash_map.h"
#include "utils/convert_utils_base.h"
#include "include/backend/visible.h"
#include "distributed/embedding_cache/cache_strategy/cache.h"

namespace mindspore {
namespace distributed {
// Define the value of an invalid step.
static constexpr size_t kInvalidStepValue = 0;
// Define the value of an invalid index.
static constexpr int kInvalidIndexValue = -1;

// The minimum valid capacity.
static constexpr size_t kMinimumCapacity = 2;

struct HashMapElement {
  // The current global step of cache prefetching operation.
  size_t step_{kInvalidStepValue};

  bool IsEmpty() const { return step_ == kInvalidStepValue; }
  bool IsExpired(size_t graph_running_step) const { return graph_running_step > step_; }
  bool StepEqual(size_t step) const { return step_ == step; }
  void set_step(size_t step) { step_ = step; }
};

// EmbeddingHashMap is used to manage the id -> index mapping of the embedding cache table on the host
// side. The cache content can be stored on the device or host side.
class BACKEND_EXPORT EmbeddingHashMap {
 public:
  using Element = typename Cache<int, int>::Element;

  explicit EmbeddingHashMap(size_t hash_capacity);

  ~EmbeddingHashMap() = default;

  // Find the insertion position (index) in the hash map for an id.
  // If the hash map capacity is insufficient, return the information of ids and indices that need to be swapped.
  int ParseData(const int id, int *const swap_out_index, int *const swap_out_ids, const size_t data_step,
                const size_t graph_running_step, size_t *const swap_out_size, bool *const need_wait_graph);

  int GetOrInsertDataUnsafe(const int key);

  // Get the global step of a element in hash map.
  size_t hash_step(const int hash_index) const;
  // Set the global step of a element in hash map.
  void set_hash_step(const int hash_index, const size_t step);

  // Get capacity of hash map.
  size_t hash_capacity() const;

  // Get index by id.
  bool GetIndex(const int id, int *index) const;

  const std::list<Element> &Export() const;

  // Reset the hash map.
  void Reset() {}

  void DumpHashMap();

 private:
  // Find the insertion position (index) in the hash map for an id.
  int FindInsertionPos(const size_t data_step, const size_t graph_running_step, bool *const need_swap,
                       bool *const need_wait_graph, int *swap_out_id);

  int InsertDataUnsafe(const int key);

  int FindPosUnsafe();

  // The hash map capacity.
  size_t hash_capacity_;

  // The hash map valid capacity(less than hash_capacity_).
  size_t valid_capacity_;

  // Record all elements in this hash map.
  std::vector<HashMapElement> hash_map_elements_;

  // The id -> index mapping.
  std::unique_ptr<Cache<int, int>> ids_to_indices_;

  // The cursor that records the current used index.
  size_t current_pos_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_HASH_MAP_H_
