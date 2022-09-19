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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_CHCHE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_CHCHE_H_

#include <cstddef>

namespace mindspore {
namespace distributed {
// Base class for implementing embedding cache, such as lru cache ,lfu cache, etc...
class EmbeddingCache {
 public:
  EmbeddingCache() = default;
  virtual ~EmbeddingCache() = default;

  // Get values which is indexed by keys at input. Input is a tensor data address from Parameter of embedding.
  virtual bool Get(void *input, size_t key_num, const void *keys, void *values) = 0;

  // Put values which is indexed by keys to input. Input is a tensor data address from Parameter of embedding.
  // When input is full, save the evicted values and keys.
  virtual bool Put(void *input, size_t key_num, const void *keys, const void *values, size_t evicted_num,
                   void *evicted_keys, void *evicted_values) = 0;

  // Check if cache is full.
  virtual bool IsFull();
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_CHCHE_H_
