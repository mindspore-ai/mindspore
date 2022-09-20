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
#include <map>
#include <string>
#include <utility>

#include "distributed/embedding_cache/embedding_store.h"
#include "distributed/embedding_cache/embedding_lru_cache.h"
#include "distributed/persistent/storage/local_file.h"

namespace mindspore {
namespace distributed {
template <typename K_T, typename V_T>
bool EmbeddingStore<K_T, V_T>::Initialize(const std::map<std::string, std::string> &storage_config) {
  value_size_ = emb_dim_ * sizeof(V_T);
  key_size_ = sizeof(K_T);
  cache_ = std::make_unique<EmbeddingLRUCache>(cache_capacity_);
  storage_ = std::make_unique<storage::LocalFile>(storage_config, key_size_, value_size_);
  return true;
}
}  // namespace distributed
}  // namespace mindspore
