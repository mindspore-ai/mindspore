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

#include "utils/graph_utils.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <stack>
#include <vector>
#include <list>
#include <string>
#include <fstream>
#include <queue>
#include <set>

#include "common/utils.h"
#include "debug/label.h"
#include "ir/func_graph.h"
#include "utils/log_adapter.h"

namespace mindspore {
std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &root, const SuccFunc &succ, const IncludeFunc &include) {
  size_t seen = NewSeenGeneration();
  std::list<AnfNodePtr> todo(1, root);
  std::unordered_map<AnfNodePtr, size_t> rank;
  std::vector<AnfNodePtr> res;

  while (!todo.empty()) {
    AnfNodePtr node = todo.back();
    if (node == nullptr || node->seen_ == seen) {
      todo.pop_back();
      continue;
    }
    if (rank.find(node) != rank.end() && rank[node] != todo.size()) {
      MS_LOG(EXCEPTION) << "Graph exists cycle, node " << node->DebugString();
    }
    rank[node] = todo.size();
    bool cont = false;
    auto incl = include(node);
    if (incl == FOLLOW) {
      auto succs = succ(node);
      for (const auto i : succs) {
        if ((i != nullptr && i->seen_ != seen)
            // Handle the case for 2 subgraphs calls each other.
            // If the ValueNodeGraph's return is already in the todo list, do not follow it.
            && !((std::find(todo.begin(), todo.end(), i) != todo.end()) && (i->func_graph() != nullptr) &&
                 (i->func_graph()->get_return() == i))) {
          todo.push_back(i);
          cont = true;
        }
      }
    } else if (incl == NOFOLLOW) {
      // do nothing
    } else if (incl == EXCLUDE) {
      node->seen_ = seen;
      todo.pop_back();
      continue;
    } else {
      MS_LOG(EXCEPTION) << "include(node) must return one of: \"follow\", \"nofollow\", \"exclude\"";
    }
    if (cont) {
      continue;
    }
    node->seen_ = seen;
    res.push_back(node);
    todo.pop_back();
  }
  return res;
}

// search the cnodes inside this graph only
std::vector<CNodePtr> BroadFirstSearchGraphCNodes(CNodePtr ret) {
  std::queue<CNodePtr> todo;
  todo.push(ret);
  std::vector<CNodePtr> sorted_nodes;
  auto seen = NewSeenGeneration();
  while (!todo.empty()) {
    CNodePtr top = todo.front();
    todo.pop();
    sorted_nodes.push_back(top);
    auto inputs = top->inputs();
    for (auto &item : inputs) {
      if (item->seen_ == seen) {
        continue;
      }

      if (item->isa<CNode>()) {
        todo.push(item->cast<CNodePtr>());
      }
      item->seen_ = seen;
    }
  }
  return sorted_nodes;
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
      auto &inputs = node->cast<CNodePtr>()->inputs();
      (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
    }
    auto graph = node->func_graph();
    if (graph->get_return() != nullptr) {
      vecs.push_back(graph->get_return());
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
      auto &inputs = node->cast<CNodePtr>()->inputs();
      (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
    }
    return vecs;
  }
}

std::vector<AnfNodePtr> SuccIncoming(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }

  if (node->isa<CNode>()) {
    auto &inputs = node->cast<CNodePtr>()->inputs();
    (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
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
    (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
  }
  return vecs;
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
