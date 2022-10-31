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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORE_H_

#include <string>
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
template <typename K, typename V>
class EmbeddingStore {
 public:
  EmbeddingStore(std::string, size_t, size_t) {}
  ~EmbeddingStore() = default;

  bool Initialize() { return true; }
  bool Finalize() { return true; }

  bool Get(const void *input, size_t key_num, const void *keys, void *values) { return true; }

  bool Get(size_t key_num, const void *keys, void *values) { return true; }

  bool Put(void *input, size_t key_num, const void *keys, const void *values) { return true; }

  bool Flush(void *input) { return true; }
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORE_H_
