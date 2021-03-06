/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "ir/graph_utils.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <stack>
#include <vector>
#include <tuple>
#include <string>
#include <fstream>
#include <deque>
#include <set>

#include "ir/func_graph.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "mindspore/ccsrc/utils/utils.h"

namespace mindspore {
std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &root, const SuccFunc &succ, const IncludeFunc &include) {
  std::vector<AnfNodePtr> res;
  if (root == nullptr) {
    return res;
  }
  size_t seen = NewSeenGeneration();
  std::deque<AnfNodePtr> todo(1024);
  todo.clear();
  todo.push_back(root);

  while (!todo.empty()) {
    AnfNodePtr node = todo.back();
    if (node->extra_seen_ == seen) {  // We use extra_seen_ as finish flag
      todo.pop_back();
      continue;
    }
    auto incl = include(node);
    if (node->seen_ == seen) {  // We use seen_ as checking flag
      todo.pop_back();
      if (incl != EXCLUDE) {
        res.push_back(node);
      }
      node->extra_seen_ = seen;
      continue;
    }
    node->seen_ = seen;
    if (incl == FOLLOW) {
      auto succs = succ(node);
      (void)std::copy_if(succs.begin(), succs.end(), std::back_inserter(todo), [seen, &todo](const AnfNodePtr &next) {
        if (next == nullptr || next->extra_seen_ == seen) {
          return false;
        }
        if (next->seen_ != seen) {
          return true;
        }
        if (next->func_graph()->get_return() == next) {
          return false;
        }
        // To dump all nodes in a circle.
        MS_LOG(ERROR) << "Graph cycle exists. Circle is: ";
        size_t pos = 0;
        auto circle_node_it = std::find(todo.begin(), todo.end(), next);
        for (; circle_node_it != todo.end(); circle_node_it++) {
          auto circle_node = *circle_node_it;
          if (circle_node->seen_ == seen) {
            MS_LOG(ERROR) << "#" << pos << ": " << circle_node->DebugString();
            pos++;
          }
        }
        MS_LOG(EXCEPTION) << "Graph cycle exists, strike node: " << next->DebugString(2);
      });
    } else if (incl > EXCLUDE) {  // Not NOFOLLOW or EXCLUDE
      MS_LOG(EXCEPTION) << "The result of include(node) must be one of: \"follow\", \"nofollow\", \"exclude\"";
    }
  }
  return res;
}

// search the cnodes inside this graph only
std::vector<CNodePtr> BroadFirstSearchGraphCNodes(const std::vector<CNodePtr> &starts) {
  std::deque<CNodePtr> todo(1024);
  todo.clear();
  todo.insert(todo.end(), starts.begin(), starts.end());
  std::vector<CNodePtr> sorted_nodes;
  auto seen = NewSeenGeneration();
  while (!todo.empty()) {
    CNodePtr top = todo.front();
    todo.pop_front();
    sorted_nodes.push_back(top);
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
  return sorted_nodes;
}

// search the cnode match the predicate inside this graph only
CNodePtr BroadFirstSearchFirstOf(const std::vector<CNodePtr> &starts, const MatchFunc &match_predicate) {
  std::deque<CNodePtr> todo(1024);
  todo.clear();
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

std::vector<FuncGraphPtr> BroadFirstSearchGraphUsed(FuncGraphPtr root) {
  std::deque<FuncGraphPtr> todo;
  todo.push_back(root);
  std::vector<FuncGraphPtr> sorted;
  auto seen = NewSeenGeneration();
  while (!todo.empty()) {
    FuncGraphPtr top = todo.front();
    todo.pop_front();
    sorted.push_back(top);
    auto used = top->func_graphs_used();
    for (auto &item : used) {
      if (item.first->seen_ == seen) {
        continue;
      }
      todo.push_back(item.first);
      item.first->seen_ = seen;
    }
  }
  return sorted;
}

// PushSuccessors push cnode inputs to a vector as successors for topo sort.
static void PushSuccessors(const CNodePtr &cnode, std::vector<AnfNodePtr> *vecs) {
  auto &inputs = cnode->inputs();
  vecs->reserve(vecs->size() + inputs.size());

  // To keep sort order from left to right in default, if kAttrTopoSortRhsFirst not set.
  auto attr_sort_rhs_first = cnode->GetAttr(kAttrTopoSortRhsFirst);
  auto sort_rhs_first =
    attr_sort_rhs_first != nullptr && attr_sort_rhs_first->isa<BoolImm>() && GetValue<bool>(attr_sort_rhs_first);
  if (sort_rhs_first) {
    vecs->insert(vecs->end(), inputs.cbegin(), inputs.cend());
  } else {
    vecs->insert(vecs->end(), inputs.crbegin(), inputs.crend());
  }
}

std::vector<AnfNodePtr> SuccDeeper(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }

  if (IsValueNode<FuncGraph>(node)) {
    auto graph = GetValueNode<FuncGraphPtr>(node);
    auto ret = graph->get_return();
    if (ret != nullptr) {
      vecs.push_back(ret);
    }
    return vecs;
  } else if (node->func_graph() != nullptr) {
    if (node->isa<CNode>()) {
      PushSuccessors(node->cast<CNodePtr>(), &vecs);
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

  if (IsValueNode<FuncGraph>(node)) {
    auto graph = GetValueNode<FuncGraphPtr>(node);
    auto ret = graph->get_return();
    if (ret != nullptr) {
      vecs.push_back(ret);
    }
    return vecs;
  } else {
    if (node->isa<CNode>()) {
      PushSuccessors(node->cast<CNodePtr>(), &vecs);
    }
    return vecs;
  }
}

std::vector<AnfNodePtr> SuccIncoming(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  auto cnode = dyn_cast<CNode>(node);
  if (cnode != nullptr) {
    PushSuccessors(cnode, &vecs);
  }
  return vecs;
}

std::vector<AnfNodePtr> SuccIncludeFV(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto &inputs = cnode->inputs();
    // Check if free variables used.
    for (const auto &input : inputs) {
      auto input_fg = GetValueNode<FuncGraphPtr>(input);
      if (input_fg) {
        for (auto &fv : input_fg->free_variables_nodes()) {
          if (fv->func_graph() == fg && fg->nodes().contains(fv)) {
            vecs.push_back(fv);
          }
        }
      }
    }
    PushSuccessors(cnode, &vecs);
  }
  return vecs;
}

const std::vector<AnfNodePtr> &GetInputs(const AnfNodePtr &node) {
  static std::vector<AnfNodePtr> empty_inputs;
  auto cnode = dyn_cast<CNode>(node);
  if (cnode != nullptr) {
    return cnode->inputs();
  }
  return empty_inputs;
}

IncludeType AlwaysInclude(const AnfNodePtr &) { return FOLLOW; }

IncludeType IncludeBelongGraph(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  if (node->func_graph() == fg) {
    return FOLLOW;
  } else {
    return EXCLUDE;
  }
}

FuncGraphIndex::FuncGraphIndex(const FuncGraphPtr &fg, const SearchFunc &search, const IncludeFunc &include) {
  MS_EXCEPTION_IF_NULL(fg);
  Acquire(fg);

  auto vec = search(fg->get_return(), include);
  for (auto &node : vec) {
    MS_EXCEPTION_IF_NULL(node);
    Acquire(node);
    if (node->func_graph() != nullptr) {
      Acquire(node->func_graph());
    }
  }
}

std::set<FuncGraphPtr> FuncGraphIndex::GetFuncGraphs(const std::string &key) {
  std::set<FuncGraphPtr> func_graphs;
  if (index_func_graph_.find(key) != index_func_graph_.end()) {
    func_graphs = index_func_graph_[key];
  }
  return func_graphs;
}

std::set<AnfNodePtr> FuncGraphIndex::GetNodes(const std::string &key) {
  if (index_node_.find(key) != index_node_.end()) {
    return index_node_[key];
  }

  return std::set<AnfNodePtr>();
}

FuncGraphPtr FuncGraphIndex::GetFirstFuncGraph(const std::string &key) {
  if (GetFuncGraphs(key).empty()) {
    return nullptr;
  }

  auto fg = *GetFuncGraphs(key).begin();
  return fg;
}

AnfNodePtr FuncGraphIndex::GetFirstNode(const std::string &key) {
  if (GetNodes(key).empty()) {
    return nullptr;
  }

  auto node = *GetNodes(key).begin();
  return node;
}

void FuncGraphIndex::Acquire(const FuncGraphPtr &key) {
  std::string name = label_manage::Label(key->debug_info());
  if (!name.empty()) {
    (void)index_func_graph_[name].insert(key);
  }
}

void FuncGraphIndex::Acquire(const AnfNodePtr &key) {
  std::string name = label_manage::Label(key->debug_info());
  if (!name.empty()) {
    (void)index_node_[name].insert(key);
  }
}
}  // namespace mindspore
