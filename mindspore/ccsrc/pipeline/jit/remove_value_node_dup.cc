/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

size_t HashOfGraph(const FuncGraphPtr &fg) {
  std::vector<AnfNodePtr> toposet = TopoSort(fg->get_return());
  MS_LOG(DEBUG) << "TopSort for:" << fg->ToString();
  std::unordered_map<AnfNodePtr, std::size_t> hashes;
  auto &params = fg->parameters();
  for (size_t i = 0; i < params.size(); i++) {
    hashes[params[i]] = std::hash<std::string>{}("param" + std::to_string(i));
  }
  for (auto node : toposet) {
    MS_EXCEPTION_IF_NULL(node);
    if (hashes.find(node) != hashes.end()) {
      continue;
    }

    std::size_t h = 0;
    if (node->isa<ValueNode>()) {
      ValueNodePtr value_node = node->cast<ValueNodePtr>();
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      if (IsValueNode<FuncGraph>(value_node)) {
        auto v_fg = value->cast<FuncGraphPtr>();
        h = value->hash();
      } else if (IsValueNode<tensor::Tensor>(value_node)) {
        // the tensor has same value has been replaced in duplicate value pass,
        // so we use the value pointer here as an identifier
        h = hash_combine(value->hash(), std::hash<Value *>{}(value.get()));
      } else {
        h = hash_combine(value->hash(), (opt::AbsOf(value_node)->hash()));
      }
    } else if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      auto &inputs = cnode->inputs();
      size_t init = 0;
      h = std::accumulate(inputs.begin(), inputs.end(), init, [&hashes](std::size_t hash, const AnfNodePtr &node_in) {
        return hash_combine(hash, hashes[node_in]);
      });
    } else if (node->isa<Parameter>()) {
      h = node->hash();
    } else {
      MS_LOG(ERROR) << "Unknow node type";
    }
    hashes[node] = h;
  }
  return hashes[fg->get_return()];
}

bool IsCNodeGraph(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }

  auto inp0 = node->cast<CNodePtr>()->input(0);
  return IsValueNode<FuncGraph>(inp0);
}

bool MergeDuplicateGraphs(const FuncGraphManagerPtr manager) {
  std::unordered_map<size_t, std::vector<FuncGraphPtr>> hash_graphs;
  std::unordered_map<FuncGraphPtr, size_t> graph_hash;
  for (auto fg : manager->func_graphs()) {
    size_t h = HashOfGraph(fg);
    graph_hash[fg] = h;
    if (hash_graphs.find(h) == hash_graphs.end()) {
      hash_graphs[h] = {fg};
    } else {
      hash_graphs[h].push_back(fg);
    }
  }
  FuncGraphPairMapEquiv equiv_graph;
  NodeMapEquiv equiv_node;
  for (auto &fg : manager->func_graphs()) {
    MS_LOG(DEBUG) << "Try Merge Graph:" << fg->ToString();
    for (auto &item : fg->nodes()) {
      if (!item->isa<CNode>()) {
        continue;
      }
      auto &inputs = item->cast<CNodePtr>()->inputs();
      for (size_t i = 0; i < inputs.size(); i++) {
        if (!inputs[i]->isa<ValueNode>()) {
          continue;
        }
        auto value_ptr = GetValueNode(inputs[i]);
        auto v_fg = value_ptr->cast<FuncGraphPtr>();
        if (v_fg == nullptr) {
          continue;
        }
        auto &fg_vec = hash_graphs[graph_hash[v_fg]];
        if (fg_vec.size() > 1) {
          if (v_fg != fg_vec[0]) {
            bool is_morphic = Isomorphic(v_fg, fg_vec[0], &equiv_graph, &equiv_node);
            if (is_morphic) {
              auto new_node = NewValueNode(fg_vec[0]);
              MS_LOG(DEBUG) << "Replace graph node :" << inputs[i]->ToString() << " with:" << new_node->ToString();
              manager->Replace(inputs[i], new_node);
            }
          }
        }
      }
    }
  }
  return true;
}
}  // namespace pipeline
}  // namespace mindspore
