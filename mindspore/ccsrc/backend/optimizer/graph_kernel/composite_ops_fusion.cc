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

#include <memory>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <map>
#include <set>
#include <queue>
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

std::vector<AnfNodePtr> DeepUsersSearch(const std::vector<AnfNodePtr> &roots, const IncludeFunc &include,
                                        const FuncGraphManagerPtr &mng) {
  std::vector<AnfNodePtr> users;
  for (auto &root : roots) {
    auto tmp = DeepUsersSearch(root, include, mng);
    users.insert(users.end(), tmp.begin(), tmp.end());
  }
  return users;
}
}  // namespace

bool IsFuse(const AnfNodePtr &node) {
  // composite fuse composite op
  if (AnfAlgo::IsGraphKernel(node)) {
    return true;
  }
  return IsBasicFuseOp(node);
}

IncludeType IncludeFusedBasicOpForward(const AnfNodePtr &cur_node, const AnfNodePtr &node) {
  if (cur_node == node) {
    return FOLLOW;
  }
  bool is_fusable = IsFuse(node);
  return is_fusable ? FOLLOW : EXCLUDE;
}

IncludeType IncludeFusedBasicOpBackward(const AnfNodePtr &cur_node, const AnfNodePtr &node) {
  if (cur_node == node) {
    return FOLLOW;
  }
  if (AnfAlgo::IsGraphKernel(node)) {
    auto cnode = node->cast<CNodePtr>();
    auto fg = GetValueNode<FuncGraphPtr>(cnode->input(kAnfPrimitiveIndex));
    auto fg_attr_val = fg->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
    MS_EXCEPTION_IF_NULL(fg_attr_val);
    auto fg_attr = GetValue<std::string>(fg_attr_val);
    if (fg_attr == kApplyMomentumOpName) {
      return FOLLOW;
    }
    return EXCLUDE;
  }
  bool is_fusable = IsBasicFuseOp(node);
  return is_fusable ? FOLLOW : EXCLUDE;
}

bool CheckCircle(const std::set<AnfNodePtr> &fused_op_set, const AnfNodePtr &check_node,
                 std::set<AnfNodePtr> *cached_unconnected_set, std::vector<AnfNodePtr> *circle_nodes) {
  if (!check_node->isa<CNode>() || !fused_op_set.count(check_node)) {
    return false;
  }
  circle_nodes->clear();

  std::set<AnfNodePtr> cached_done_set;
  auto cnode = check_node->cast<CNodePtr>();
  const auto &inputs = cnode->inputs();
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
          for (auto it : cnode_ptr->inputs()) {
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

std::vector<AnfNodePtr> RemoveCircle(const std::vector<AnfNodePtr> &fused_op, bool is_backward) {
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
    bool has_circle = CheckCircle(fused_op_set, *iter, &cached_unconnected_set, &circle_nodes);
    // delete the circle node and the node which depend on the circle node in fused op
    if (has_circle) {
      auto mng = (*iter)->func_graph()->manager();
      std::vector<AnfNodePtr> erase_nodes;
      if (is_backward) {
        erase_nodes = DeepUsersSearch(circle_nodes, include, mng);
      } else {
        erase_nodes = DeepLinkedGraphSearch(circle_nodes, include);
      }
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

std::vector<AnfNodePtr> FindFuseCNodes(const CNodePtr &cnode) {
  auto func_graph = cnode->func_graph();
  auto mng = func_graph->manager();
  // Search fusable nodes according input direction.
  auto include_func_forward = std::bind(IncludeFusedBasicOpForward, cnode, std::placeholders::_1);
  auto used_nodes = DeepLinkedGraphSearch(cnode, include_func_forward);
  std::reverse(used_nodes.begin(), used_nodes.end());
  // Search fusable nodes according output direction.
  auto include_func_backward = std::bind(IncludeFusedBasicOpBackward, cnode, std::placeholders::_1);
  auto user_nodes = DeepUsersSearch(cnode, include_func_backward, mng);

  used_nodes.insert(used_nodes.end(), user_nodes.begin() + 1, user_nodes.end());
  if (used_nodes.size() > 1) {
    used_nodes = RemoveCircle(used_nodes);
  }
  TopoSortForNodeList(&used_nodes);
  return used_nodes;
}

bool FuseCompositeOps(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool changed = false;
  auto &todos = kernel_graph->execution_order();
  for (auto iter = todos.cbegin(); iter != todos.cend(); ++iter) {
    auto node = *iter;
    if (!AnfAlgo::IsGraphKernel(node) || !kernel_graph->nodes().contains(node)) {
      continue;
    }

    auto origin_fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
    auto fg_attr = origin_fg->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
    if (fg_attr != nullptr) {
      auto fg_name = GetValue<std::string>(fg_attr);
      if (graph_kernel_black_list.count(fg_name) != 0) {
        continue;
      }
    }

    auto fuse_nodes = FindFuseCNodes(node);
    if (fuse_nodes.size() <= 1) {
      continue;
    }
    changed = true;

    FuseNodesToSubGraph(fuse_nodes, kernel_graph, "");
  }
  return changed;
}

bool CompositeOpsFusion::Run(const FuncGraphPtr &func_graph) {
  return FuseCompositeOps(std::dynamic_pointer_cast<session::KernelGraph>(func_graph));
}
}  // namespace opt
}  // namespace mindspore
