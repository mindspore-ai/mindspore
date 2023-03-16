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
#include "utils/hash_map.h"
#include "utils/convert_utils_base.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
// Define the value of an invalid step.
static constexpr size_t INVALID_STEP_VALUE = 0;
// Define the value of an invalid index.
static constexpr int INVALID_INDEX_VALUE = -1;

struct HashMapElement {
  int id_{INVALID_INDEX_VALUE};
  // The current global step of cache prefetching operation.
  size_t step_{INVALID_STEP_VALUE};

  bool IsEmpty() const { return step_ == INVALID_STEP_VALUE; }
  bool IsExpired(size_t graph_running_step) const { return graph_running_step > step_; }
  bool StepEqual(size_t step) const { return step_ == step; }
  void set_id(int id) { id_ = id; }
  void set_step(size_t step) { step_ = step; }
};

// EmbeddingHashMap is used to manage the id -> index mapping of the embedding cache table on the host
// side. The cache content can be stored on the device or host side.
class BACKEND_EXPORT EmbeddingHashMap {
 public:
  EmbeddingHashMap(size_t hash_count, size_t hash_capacity);

  ~EmbeddingHashMap() = default;

  // Find the insertion position (index) in the hash map for an id.
  // If the hash map capacity is insufficient, return the information of ids and indices that need to be swapped.
  int ParseData(const int id, int *const swap_out_index, int *const swap_out_ids, const size_t data_step,
                const size_t graph_running_step, size_t *const swap_out_size, bool *const need_wait_graph);

  // Get the global step of a element in hash map.
  size_t hash_step(const int hash_index) const;
  // Set the global step of a element in hash map.
  void set_hash_step(const int hash_index, const size_t step);
  // Get the id -> index mapping.
  const mindspore::HashMap<int, int> &hash_id_to_index() const;
  // Get capacity of hash map.
  size_t hash_capacity() const;
  // Reset the hash map.
  void Reset();

  void DumpHashMap();

 private:
  // Find the insertion position (index) in the hash map for an id.
  int FindInsertionPos(const size_t data_step, const size_t graph_running_step, bool *const need_swap,
                       bool *const need_wait_graph);

  // Statistics on the usage of hash map capacity.
  size_t hash_count_;

  // The hash map capacity.
  size_t hash_capacity_;

  // Record all elements in this hash map.
  std::vector<HashMapElement> hash_map_elements_;

  // The id -> index mapping.
  mindspore::HashMap<int, int> hash_id_to_index_;

  // The cursor that records the current slot.
  size_t current_pos_;
  // The cursor that records the start position of current_pos_.
  size_t current_batch_start_pos_;

  // The number of ids which need to wait for the calculation graph to finish executing the current step and need be
  // swapped out.
  size_t graph_running_index_num_;
  // The index in array 'graph_running_index_', and the value on this index is the hash index for new id,
  // but need to wait for the calculation graph to finish executing the current step and swap out the expired data.
  size_t graph_running_index_pos_;
  // Record the index information of the feature id that needs to be swapped out after the calculation graph finishes
  // executing the current step.
  std::unique_ptr<int[]> graph_running_index_;

  // The flag indicates hash map is full.
  bool expired_element_full_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_HASH_MAP_H_
