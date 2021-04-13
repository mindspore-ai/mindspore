/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_PS_CACHE_EMBEDDING_HASH_MAP_H_
#define MINDSPORE_CCSRC_PS_PS_CACHE_EMBEDDING_HASH_MAP_H_

#include <math.h>
#include <utility>
#include <memory>
#include <vector>
#include <unordered_map>
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace ps {
static const size_t INVALID_STEP_VALUE = 0;
static const int INVALID_INDEX_VALUE = -1;

struct HashMapElement {
  int id_{INVALID_INDEX_VALUE};
  size_t step_{INVALID_STEP_VALUE};
  bool IsEmpty() const { return step_ == INVALID_STEP_VALUE; }
  bool IsExpired(size_t graph_running_step) const { return graph_running_step > step_; }
  bool IsStep(size_t step) const { return step_ == step; }
  void set_id(int id) { id_ = id; }
  void set_step(size_t step) { step_ = step; }
};

// Hash table is held in device, HashMap is used to manage hash table in host.
class EmbeddingHashMap {
 public:
  EmbeddingHashMap(size_t hash_count, size_t hash_capacity)
      : hash_count_(hash_count),
        hash_capacity_(hash_capacity),
        current_pos_(0),
        current_batch_start_pos_(0),
        graph_running_index_num_(0),
        graph_running_index_pos_(0),
        expired_element_full_(false) {
    hash_map_elements_.resize(hash_capacity);
    // In multi-device mode, embedding table are distributed on different devices by ID interval,
    // and IDs outside the range of local device will use the front and back positions of the table,
    // the positions are reserved for this.
    hash_map_elements_.front().set_step(SIZE_MAX);
    hash_map_elements_.back().set_step(SIZE_MAX);
    graph_running_index_ = std::make_unique<int[]>(hash_capacity);
  }
  virtual ~EmbeddingHashMap() = default;
  int ParseData(const int id, int *const swap_out_index, int *const swap_out_ids, const size_t data_step,
                const size_t graph_running_step, size_t *const swap_out_size, bool *const need_wait_graph);
  size_t hash_step(const int hash_index) const { return hash_map_elements_[hash_index].step_; }
  void set_hash_step(const int hash_index, const size_t step) { hash_map_elements_[hash_index].set_step(step); }
  const std::unordered_map<int, int> &hash_id_to_index() const { return hash_id_to_index_; }
  size_t hash_capacity() const { return hash_capacity_; }
  void DumpHashMap();
  void Reset();

 private:
  int FindInsertionPos(const size_t data_step, const size_t graph_running_step, bool *const need_swap,
                       bool *const need_wait_graph);
  size_t hash_count_;
  size_t hash_capacity_;
  std::vector<HashMapElement> hash_map_elements_;
  std::unordered_map<int, int> hash_id_to_index_;
  size_t current_pos_;
  size_t current_batch_start_pos_;
  size_t graph_running_index_num_;
  size_t graph_running_index_pos_;
  std::unique_ptr<int[]> graph_running_index_;
  bool expired_element_full_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PS_CACHE_EMBEDDING_HASH_MAP_H_
