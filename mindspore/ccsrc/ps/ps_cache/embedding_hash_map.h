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
  void set_id(int id) { id_ = id; }
  void set_step(size_t step) { step_ = step; }
};

// Hash table is held in device, HashMap is used to manage hash table in host.
class EmbeddingHashMap {
 public:
  EmbeddingHashMap(size_t hash_count, size_t hash_capacity) : hash_count_(hash_count), hash_capacity_(hash_capacity) {
    hash_map_elements_.resize(hash_capacity);
  }
  virtual ~EmbeddingHashMap() = default;
  int ParseData(const int id, int *swap_out_index, int *swap_out_ids, const size_t data_step,
                const size_t graph_running_step, size_t *swap_out_size);
  size_t hash_step(const int hash_index) const { return hash_map_elements_[hash_index].step_; }
  void set_hash_step(const int hash_index, const size_t step) { hash_map_elements_[hash_index].set_step(step); }
  const std::unordered_map<int, int> &hash_id_to_index() const { return hash_id_to_index_; }
  size_t hash_capacity() const { return hash_capacity_; }
  void DumpHashMap();

 private:
  int Hash(const int id) { return static_cast<int>((0.6180339 * id - std::floor(0.6180339 * id)) * hash_capacity_); }
  bool NeedSwap() const { return hash_count_ > FloatToSize(hash_capacity_ * 0.9); }
  size_t hash_count_;
  size_t hash_capacity_;
  std::vector<HashMapElement> hash_map_elements_;
  std::unordered_map<int, int> hash_id_to_index_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PS_CACHE_EMBEDDING_HASH_MAP_H_
