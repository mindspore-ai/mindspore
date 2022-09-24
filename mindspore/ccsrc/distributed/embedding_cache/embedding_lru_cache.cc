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

#include "distributed/embedding_cache/embedding_lru_cache.h"

namespace mindspore {
namespace distributed {
template <typename K, typename V>
bool EmbeddingLRUCache<K, V>::Initialize() {
  keys_lru_cache_ = std::make_unique<LRUCache<K, size_t>>(capacity_);
  return true;
}

template class EmbeddingLRUCache<size_t, float>;
template class EmbeddingLRUCache<size_t, double>;
template class EmbeddingLRUCache<size_t, int64_t>;
template class EmbeddingLRUCache<size_t, size_t>;
}  // namespace distributed
}  // namespace mindspore
