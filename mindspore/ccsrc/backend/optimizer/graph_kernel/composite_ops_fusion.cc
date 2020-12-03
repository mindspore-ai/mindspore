/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/composite_ops_fusion.h"

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "frontend/operator/ops.h"
#include "utils/utils.h"
#include "utils/ordered_set.h"
#include "utils/ordered_map.h"
#include "ir/graph_utils.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "vm/segment_runner.h"
#include "debug/draw.h"
#include "debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/optimizer/pass/getitem_tuple.h"

namespace mindspore {
namespace opt {
namespace {
std::vector<AnfNodePtr> DeepLinkedGraphSearch(const std::vector<AnfNodePtr> &roots, const IncludeFunc &include) {
  std::vector<AnfNodePtr> inputs;
  for (auto &root : roots) {
    auto tmp = DeepLinkedGraphSearch(root, include);
    inputs.insert(inputs.end(), tmp.begin(), tmp.end());
  }
  return inputs;
}
}  // namespace

bool CheckCircle(const std::set<AnfNodePtr> &fused_op_set, const AnfNodePtr &check_node,
                 std::set<AnfNodePtr> *cached_unconnected_set, std::vector<AnfNodePtr> *circle_nodes,
                 const std::multimap<AnfNodePtr, std::pair<AnfNodePtr, AnfNodePtr>> &depend_prior) {
  if (!check_node->isa<CNode>() || !fused_op_set.count(check_node)) {
    return false;
  }
  circle_nodes->clear();

  auto InputEdges = [&depend_prior](CNodePtr cnode) {
    std::set<AnfNodePtr> edges;
    auto range = depend_prior.equal_range(cnode);
    for (auto iter = range.first; iter != range.second; ++iter) {
      edges.insert(iter->second.first);
    }
    auto inputs = cnode->inputs();
    for (auto input : inputs) {
      edges.insert(input);
    }
    return edges;
  };

  std::set<AnfNodePtr> cached_done_set;
  auto cnode = check_node->cast<CNodePtr>();
  const auto &inputs = InputEdges(cnode);
  // there is a input not in fused_op_set, but the input depends on the fused_op_set
  for (auto input : inputs) {
    if (input->isa<CNode>() && !fused_op_set.count(input)) {
      bool has_circle = false;
      std::set<AnfNodePtr> done;
      std::vector<AnfNodePtr> todos = {input};
      while (!todos.empty()) {
        auto node = todos.back();
        todos.pop_back();
        if (done.count(node) || cached_unconnected_set->count(node) || cached_done_set.count(node)) {
          continue;
        }

        done.insert(node);
        if (fused_op_set.count(node)) {
          has_circle = true;
          circle_nodes->push_back(node);
          continue;
        }

        if (node->isa<CNode>()) {
          auto cnode_ptr = node->cast<CNodePtr>();
          for (auto it : InputEdges(cnode_ptr)) {
            if (it->isa<CNode>()) {
              todos.push_back(it);
            }
          }
        }
      }

      if (has_circle) {
        cached_done_set.insert(done.begin(), done.end());
      } else {
        cached_unconnected_set->insert(done.begin(), done.end());
      }
      done.clear();
    }
  }

  return !circle_nodes->empty();
}

AnfNodePtrList RemoveCircle(const std::vector<AnfNodePtr> &fused_op,
                            const std::multimap<AnfNodePtr, std::pair<AnfNodePtr, AnfNodePtr>> &depend_prior) {
  std::set<AnfNodePtr> cached_unconnected_set;
  std::set<AnfNodePtr> fused_op_set(fused_op.begin(), fused_op.end());
  auto include = [&fused_op_set](const AnfNodePtr &node) {
    if (fused_op_set.count(node)) {
      return FOLLOW;
    }
    return EXCLUDE;
  };

  std::vector<AnfNodePtr> circle_nodes;
  for (auto iter = fused_op.rbegin(); iter != fused_op.rend(); ++iter) {
    circle_nodes.clear();
    bool has_circle = CheckCircle(fused_op_set, *iter, &cached_unconnected_set, &circle_nodes, depend_prior);
    // delete the circle node and the node which depend on the circle node in fused op
    if (has_circle) {
      std::vector<AnfNodePtr> erase_nodes;
      erase_nodes = DeepLinkedGraphSearch(circle_nodes, include);
      for (auto erase_node : erase_nodes) {
        fused_op_set.erase(erase_node);
      }
    }
  }

  std::vector<AnfNodePtr> res;
  for (auto node : fused_op) {
    if (fused_op_set.count(node)) {
      res.push_back(node);
    }
  }
  return res;
}

void TopoSortForNodeList(std::vector<AnfNodePtr> *lst) {
  if (lst->size() < 2) {
    return;
  }

  std::vector<AnfNodePtr> res;
  std::set<AnfNodePtr> node_sets(lst->begin(), lst->end());
  OrderedMap<AnfNodePtr, std::set<AnfNodePtr>> ins;
  OrderedMap<AnfNodePtr, OrderedSet<AnfNodePtr>> outs;
  std::queue<AnfNodePtr> q;
  for (auto node : *lst) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (auto input : cnode->inputs()) {
      if (!node_sets.count(input)) {
        continue;
      }
      // out_degree
      outs[input].insert(node);
      // in_degree
      ins[node].insert(input);
    }
    if (!ins.count(node)) {
      ins[node] = {};
    }
  }

  for (auto p : ins) {
    if (p.second.size() == 0) {
      q.push(p.first);
    }
  }

  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    res.push_back(node);
    if (!outs.count(node)) {
      continue;
    }
    for (auto out : outs[node]) {
      if (!ins.count(out)) {
        continue;
      }
      ins[out].erase(node);
      if (ins[out].size() == 0) {
        q.push(out);
      }
    }
  }

  lst->assign(res.begin(), res.end());
}
}  // namespace opt
}  // namespace mindspore
