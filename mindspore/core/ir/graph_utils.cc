/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "ir/graph_utils.h"
#include <utility>
#include <deque>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"

namespace mindspore {
// Dump the circle from the strike node `next`.
static size_t DumpSortingCircleList(const std::deque<AnfNodePtr> &todo, const AnfNodePtr &next, SeenNum seen) {
  size_t pos = 0;
  auto circle_node_it = std::find(todo.begin(), todo.end(), next);
  for (; circle_node_it != todo.end(); circle_node_it++) {
    auto circle_node = *circle_node_it;
    if (circle_node->seen_ == seen) {
      MS_LOG(ERROR) << "#" << pos << ": " << circle_node->DebugString();
      pos++;
    }
  }
  return pos;
}

std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &root, const SuccFunc &succ, const IncludeFunc &include) {
  constexpr size_t kVecReserve = 64;
  std::vector<AnfNodePtr> res;
  if (root == nullptr) {
    return res;
  }
  res.reserve(kVecReserve);
  auto seen = NewSeenGeneration();
  std::deque<AnfNodePtr> todo;
  (void)todo.emplace_back(root);
  while (!todo.empty()) {
    AnfNodePtr &node = todo.back();
    if (node->extra_seen_ == seen) {  // We use extra_seen_ as finish flag
      todo.pop_back();
      continue;
    }
    auto incl = include(node);
    if (node->seen_ == seen) {  // We use seen_ as checking flag
      node->extra_seen_ = seen;
      if (incl != EXCLUDE) {
        (void)res.emplace_back(std::move(node));
      }
      todo.pop_back();
      continue;
    }
    node->seen_ = seen;
    if (incl == FOLLOW) {
      for (auto &next : succ(node)) {
        if (next == nullptr || next->extra_seen_ == seen) {
          continue;
        }
        if (next->seen_ != seen) {
          (void)todo.emplace_back(std::move(next));
          continue;
        }
        auto fg = next->func_graph();
        if (fg != nullptr && fg->return_node() == next) {
          continue;
        }
        // To dump all nodes in a circle.
        MS_LOG(ERROR) << "Graph cycle exists. Circle is: ";
        auto circle_len = DumpSortingCircleList(todo, next, seen);
        MS_LOG(EXCEPTION) << "Graph cycle exists, size: " << circle_len << ", strike node: " << next->DebugString(2);
      }
    } else if (incl > EXCLUDE) {  // Not NOFOLLOW or EXCLUDE
      MS_LOG(EXCEPTION) << "The result of include(node) must be one of: \"follow\", \"nofollow\", \"exclude\"";
    }
  }
  return res;
}

// Search the cnodes inside this graph only.
std::vector<CNodePtr> BroadFirstSearchGraphCNodes(const CNodePtr &start) {
  constexpr size_t kVecReserve = 64;
  std::vector<CNodePtr> vec;
  vec.reserve(kVecReserve);
  auto seen = NewSeenGeneration();
  start->seen_ = seen;
  (void)vec.emplace_back(start);
  for (size_t i = 0; i < vec.size(); ++i) {
    CNodePtr &node = vec[i];
    auto &inputs = node->inputs();
    for (auto &input : inputs) {
      if (input->seen_ == seen) {
        continue;
      }
      input->seen_ = seen;
      auto input_cnode = input->cast<CNodePtr>();
      if (input_cnode != nullptr) {
        (void)vec.emplace_back(std::move(input_cnode));
      }
    }
  }
  return vec;
}

// search the cnode match the predicate inside this graph only
CNodePtr BroadFirstSearchFirstOf(const std::vector<CNodePtr> &starts, const MatchFunc &match_predicate) {
  std::deque<CNodePtr> todo;
  todo.insert(todo.end(), starts.begin(), starts.end());
  auto seen = NewSeenGeneration();
  while (!todo.empty()) {
    CNodePtr top = todo.front();
    todo.pop_front();
    if (match_predicate(top)) {
      return top;
    }
    auto inputs = top->inputs();
    for (auto &item : inputs) {
      if (item->seen_ == seen) {
        continue;
      }

      if (item->isa<CNode>()) {
        todo.push_back(item->cast<CNodePtr>());
      }
      item->seen_ = seen;
    }
  }
  return nullptr;
}

std::vector<FuncGraphPtr> BroadFirstSearchGraphUsed(const FuncGraphPtr &root) {
  std::vector<FuncGraphPtr> todo;
  todo.push_back(root);
  auto seen = NewSeenGeneration();
  size_t top_idx = 0;
  while (top_idx < todo.size()) {
    FuncGraphPtr top = todo[top_idx];
    top_idx++;
    auto used = top->func_graphs_used();
    for (auto &item : used) {
      if (item.first->seen_ == seen) {
        continue;
      }
      todo.push_back(item.first);
      item.first->seen_ = seen;
    }
  }
  return todo;
}

// To get CNode inputs to a vector as successors for TopoSort().
static void FetchCNodeSuccessors(const CNodePtr &cnode, std::vector<AnfNodePtr> *vecs) {
  auto &inputs = cnode->inputs();
  vecs->reserve(vecs->size() + inputs.size());

  // To keep sort order from left to right in default, if kAttrTopoSortRhsFirst not set.
  auto attr_sort_rhs_first = cnode->GetAttr(kAttrTopoSortRhsFirst);
  auto sort_rhs_first =
    attr_sort_rhs_first != nullptr && attr_sort_rhs_first->isa<BoolImm>() && GetValue<bool>(attr_sort_rhs_first);
  if (sort_rhs_first) {
    (void)vecs->insert(vecs->end(), inputs.cbegin(), inputs.cend());
  } else {
    (void)vecs->insert(vecs->end(), inputs.crbegin(), inputs.crend());
  }
}

std::vector<AnfNodePtr> SuccDeeper(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }

  auto graph = GetValuePtr<FuncGraph>(node);
  if (graph != nullptr) {
    auto &ret = graph->return_node();
    if (ret != nullptr) {
      vecs.push_back(ret);
    }
    return vecs;
  } else if (node->func_graph() != nullptr) {
    if (node->isa<CNode>()) {
      FetchCNodeSuccessors(node->cast<CNodePtr>(), &vecs);
    }
    return vecs;
  }

  return vecs;
}

std::vector<AnfNodePtr> SuccDeeperSimple(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }

  auto graph = GetValuePtr<FuncGraph>(node);
  if (graph != nullptr) {
    auto &ret = graph->return_node();
    if (ret != nullptr) {
      vecs.push_back(ret);
    }
  } else if (node->isa<CNode>()) {
    FetchCNodeSuccessors(node->cast<CNodePtr>(), &vecs);
  }
  return vecs;
}

std::vector<AnfNodePtr> SuccIncoming(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  auto cnode = dyn_cast<CNode>(node);
  if (cnode != nullptr) {
    FetchCNodeSuccessors(cnode, &vecs);
  }
  return vecs;
}

std::vector<AnfNodePtr> SuccIncludeFV(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr) {
    return {};
  }
  std::vector<AnfNodePtr> vecs;
  const auto &inputs = cnode->inputs();
  // Check if free variables used.
  for (const auto &input : inputs) {
    auto input_fg = GetValuePtr<FuncGraph>(input);
    if (input_fg != nullptr) {
      for (auto &fv : input_fg->free_variables_nodes()) {
        if (fv->func_graph() == fg && fg->nodes().contains(fv)) {
          vecs.push_back(fv);
        }
      }
    }
  }
  FetchCNodeSuccessors(cnode, &vecs);
  return vecs;
}

std::vector<AnfNodePtr> SuccWithFilter(const GraphFilterFunc &graph_filter, const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }

  auto graph = GetValueNode<FuncGraphPtr>(node);
  if (graph != nullptr) {
    if (graph_filter != nullptr && graph_filter(graph)) {
      return vecs;
    }
    auto &ret = graph->return_node();
    if (ret != nullptr) {
      vecs.push_back(ret);
    }
  } else if (node->isa<CNode>()) {
    FetchCNodeSuccessors(node->cast<CNodePtr>(), &vecs);
  }
  return vecs;
}

const std::vector<AnfNodePtr> &GetInputs(const AnfNodePtr &node) {
  static std::vector<AnfNodePtr> empty_inputs;
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode != nullptr) {
    return cnode->inputs();
  }
  return empty_inputs;
}

IncludeType IncludeBelongGraph(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  if (node->func_graph() == fg) {
    return FOLLOW;
  } else {
    return EXCLUDE;
  }
}
}  // namespace mindspore
