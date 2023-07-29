/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/remove_value_node_dup.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "include/common/utils/cse.h"
#include "utils/log_adapter.h"
#include "utils/hashing.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace pipeline {
static inline bool IsSameValue(const Value *v1, const Value *v2) {
  if (v1->isa<tensor::Tensor>() && v2->isa<tensor::Tensor>()) {
    return static_cast<const tensor::Tensor *>(v1)->ValueEqual(*(static_cast<const tensor::Tensor *>(v2)));
  }
  return *v1 == *v2;
}

void TryToDoReplace(FuncGraphManager *const manager, const AnfNodePtr &node, HashCache *const hash_cache,
                    HashValue *const hash_value) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(hash_cache);

  if (IsValueNode<FuncGraph>(node)) {
    return;
  }
  auto to_check_value = GetValuePtr(node);
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
    auto existed_value = GetValuePtr(v);
    MS_EXCEPTION_IF_NULL(existed_value);
    if (IsSameValue(existed_value, to_check_value)) {
      (void)manager->Replace(node, v);
      return;
    }
  }
  // Meet for the first time, append node to bucket.
  (void)bucket.emplace_back(node);
}
}  // namespace pipeline
}  // namespace mindspore
