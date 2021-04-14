/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/basic_ops_fusion.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>

#include "base/core_ops.h"
#include "ir/graph_utils.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "vm/segment_runner.h"
#include "debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"
#include "backend/optimizer/graph_kernel/composite_ops_fusion.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/optimizer/pass/getitem_tuple.h"

namespace mindspore {
namespace opt {
namespace {
bool IsFusibleOp(const AnfNodePtr &node) {
#if ENABLE_D
  const std::set<std::string> graph_kernel_black_list = {"BNTrainingUpdateSum", "ApplyMomentum", "LayerNormForward",
                                                         "LambNextMV", "LambUpdateWithLR"};
  if (AnfAlgo::IsGraphKernel(node)) {
    auto fg_attr = AnfAlgo::GetCNodeFuncGraphPtr(node)->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
    if (fg_attr != nullptr) {
      return graph_kernel_black_list.count(GetValue<std::string>(fg_attr)) == 0;
    }
  }
#endif
  return IsBasicFuseOp(node) || AnfAlgo::IsGraphKernel(node);
}

IncludeType IncludeFusedBasicOpForward(const AnfNodePtr &cur_node, const AnfNodePtr &node) {
  if (cur_node == node) {
    return FOLLOW;
  }
  if (IsFusibleOp(node)) {
    return FOLLOW;
  }
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    auto prev_node = node->cast<CNodePtr>()->input(kRealInputNodeIndexInTupleGetItem);
    if (AnfAlgo::IsGraphKernel(prev_node)) {
      return FOLLOW;
    }
  }
  return EXCLUDE;
}

// The GetItem node should be fused with its real input and users.
// If its real input is not in the fuse_list, the GetItem should be excluded.
AnfNodePtrList RemoveWildGetitem(const AnfNodePtrList &fused_op) {
  if (fused_op.empty()) return AnfNodePtrList();
  std::set<AnfNodePtr> fused_op_set(fused_op.begin(), fused_op.end());
  auto check_include = [&fused_op_set](const AnfNodePtr &node) { return fused_op_set.count(node) ? FOLLOW : EXCLUDE; };

  auto mng = fused_op[0]->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = true;
  while (changed) {
    changed = false;
    AnfNodePtrList remove_list;
    for (auto getitem : fused_op_set) {
      if (!AnfAlgo::CheckPrimitiveType(getitem, prim::kPrimTupleGetItem)) continue;

      // GetItem should be fused with its real input.
      auto prev_node = getitem->cast<CNodePtr>()->input(kRealInputNodeIndexInTupleGetItem);
      if (check_include(prev_node) == EXCLUDE) {
        remove_list.push_back(getitem);
        break;
      }

      // GetItem should be fused with its all users.
      const auto &users = mng->node_users()[getitem];
      if (std::any_of(users.begin(), users.end(), [check_include](const std::pair<AnfNodePtr, int> &user) {
            return check_include(user.first) == EXCLUDE;
          })) {
        remove_list = DeepLinkedGraphSearch(getitem, check_include);
        break;
      }
    }
    if (!remove_list.empty()) {
      for (auto node : remove_list) {
        fused_op_set.erase(node);
      }
      changed = true;
    }
  }

  // keep the original order of fused_op.
  AnfNodePtrList result;
  for (auto node : fused_op) {
    if (fused_op_set.count(node)) {
      result.push_back(node);
    }
  }
  return result;
}

std::vector<AnfNodePtr> FindFuseCNodes(const CNodePtr &cnode,
                                       const std::multimap<AnfNodePtr, std::pair<AnfNodePtr, AnfNodePtr>> &dep_pri) {
  // Search fusable nodes according input direction.
  auto include_func_forward = std::bind(IncludeFusedBasicOpForward, cnode, std::placeholders::_1);
  auto used_nodes = DeepLinkedGraphSearch(cnode, include_func_forward);
  if (used_nodes.size() > 1) {
    used_nodes = RemoveCircle(used_nodes, dep_pri);
  }
  used_nodes = RemoveWildGetitem(used_nodes);
  TopoSortForNodeList(&used_nodes);
  return used_nodes;
}

bool FuseBasicOps(const FuncGraphPtr &kernel_graph, const std::vector<AnfNodePtr> &todos,
                  std::unordered_set<AnfNodePtr> *fused_ops) {
  bool changed = false;
  auto mng = kernel_graph->manager();

  // depend_prior[depend] = pair(prior, controlDependNode)
  std::multimap<AnfNodePtr, std::pair<AnfNodePtr, AnfNodePtr>> depend_prior;
  InitDependPrior(todos, &depend_prior);

  for (auto iter = todos.cbegin(); iter != todos.cend(); ++iter) {
    auto node = (*iter)->cast<CNodePtr>();
    if (node == nullptr || IsKeepBasicNode(node) || fused_ops->count(node)) {
      continue;
    }
    bool is_fusible_op = IsFusibleOp(node);
    if (!is_fusible_op || !kernel_graph->nodes().contains(node)) {
      continue;
    }

    auto fuse_nodes = FindFuseCNodes(node, depend_prior);
    if (fuse_nodes.empty()) {
      continue;
    }

    if (fuse_nodes.size() == 1) {
      // Do not fuse a single GraphKernel again.
      // Do not fuse a single Assign.
      if (AnfAlgo::IsGraphKernel(fuse_nodes[0]) || IsPrimitiveCNode(fuse_nodes[0], prim::kPrimAssign)) {
        continue;
      }
    }

    changed = true;
    fused_ops->insert(fuse_nodes.begin(), fuse_nodes.end());
    AnfNodePtr fused_new_node;
    AnfNodePtrList old_outputs;
    std::tie(fused_new_node, old_outputs) = FuseNodesToSubGraph(fuse_nodes, kernel_graph, "fusion");
    ReplaceNewFuseCNodeForDependPrior(&depend_prior, fused_new_node, old_outputs);
  }
  std::dynamic_pointer_cast<session::KernelGraph>(kernel_graph)->SetExecOrderByDefault();
  return changed;
}
}  // namespace

bool FuseBasicOps(const FuncGraphPtr &func_graph) {
  std::unordered_set<AnfNodePtr> fused_ops;
  auto todos = TopoSort(func_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  return FuseBasicOps(func_graph, todos, &fused_ops);
}

void EliminateGetitem(const FuncGraphPtr &func_graph) {
  std::shared_ptr<Pass> eliminate_getitem_pass = std::make_shared<opt::GetitemTuple>();
  auto todos = TopoSort(func_graph->get_return());
  for (auto node : todos) {
    if (AnfAlgo::IsGraphKernel(node)) {
      eliminate_getitem_pass->Run(AnfAlgo::GetCNodeFuncGraphPtr(node));
    }
  }
}

bool BasicOpsFusion::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool changed = FuseBasicOps(func_graph);
  if (changed) {
    EliminateGetitem(func_graph);
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
