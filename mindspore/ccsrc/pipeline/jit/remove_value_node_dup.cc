/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <string>

#include "pipeline/jit/remove_value_node_dup.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "frontend/optimizer/cse.h"
#include "utils/log_adapter.h"
#include "utils/hashing.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace pipeline {
void TryToDoReplace(FuncGraphManager *const manager, const AnfNodePtr &node, HashCache *const hash_cache,
                    HashValue *const hash_value) {
  const auto &to_check_value = GetValueNode(node);
  MS_EXCEPTION_IF_NULL(to_check_value);

  // Calculate hash value.
  size_t h;
  auto hash_iter = hash_value->find(node);
  if (hash_iter == hash_value->end()) {
    h = hash_combine(to_check_value->hash(), (opt::AbsOf(node)->hash()));
    (*hash_value)[node] = h;
  } else {
    h = hash_iter->second;
  }

  auto bucket_iter = hash_cache->find(h);
  if (bucket_iter == hash_cache->end()) {
    // Meet for the first time, add bucket.
    (*hash_cache)[h] = {node};
    return;
  }

  auto &bucket = bucket_iter->second;
  // Check if need to replace node with value node already met.
  for (const auto &v : bucket) {
    // Already met and cached.
    if (v == node) {
      return;
    }
    const auto &existed_value = GetValueNode(v);
    MS_EXCEPTION_IF_NULL(existed_value);
    auto equal = [&]() -> bool {
      if (existed_value->isa<tensor::Tensor>() && to_check_value->isa<tensor::Tensor>()) {
        return existed_value->cast<tensor::TensorPtr>()->ValueEqual(*(to_check_value->cast<tensor::TensorPtr>()));
      }
      return *existed_value == *to_check_value;
    };
    if (equal()) {
      (void)manager->Replace(node, v);
      return;
    }
  }

  // Meet for the first time, append node to bucket.
  bucket.emplace_back(node);
}
}  // namespace pipeline
}  // namespace mindspore
