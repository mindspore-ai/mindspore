/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/merge_recompute_call_nodes.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <memory>
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr char MERGE_RECOMPUTE_CALL_NODES_RUN_ONCE_ONLY[] = "merge_recompute_call_nodes_run_once_only";

CNodePtr GetRecomputeCallNode(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return nullptr;
  }
  const auto &tuple_get_item_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node->cast<CNodePtr>());
  auto k_fg_caller = node->cast<CNodePtr>()->input(kIndex1);
  MS_EXCEPTION_IF_NULL(k_fg_caller);
  auto recompute_call_node = k_fg_caller->cast<CNodePtr>();
  if (recompute_call_node && recompute_call_node->HasAttr(kAddedRecomputeDependAttr)) {
    return recompute_call_node;
  }
  return nullptr;
}

AnfNodePtr GetDependRelyNode(const CNodePtr &cnode) {
  AnfNodePtr rely_node = nullptr;
  for (size_t i = kIndex1; i < cnode->size(); ++i) {
    if (!IsPrimitiveCNode(cnode->input(i), prim::kPrimDepend)) {
      continue;
    }
    auto depend_cnode = cnode->input(i)->cast<CNodePtr>();
    if (!rely_node) {
      rely_node = depend_cnode->input(kIndex2);
      MS_EXCEPTION_IF_NULL(rely_node);
    } else if (rely_node != depend_cnode->input(kIndex2)) {
      MS_LOG(WARNING) << "For recompute call node:" << cnode->DebugString()
                      << " does not rely on the same control edge,"
                         " thus not doing recompute comm overlap.";
      return nullptr;
    }
  }
  return rely_node;
}

void ReplaceDependRelyNode(const CNodePtr &cnode, const AnfNodePtr &pre_rely_node) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  const auto &manager = cnode->func_graph()->manager();
  for (size_t i = kIndex1; i < cnode->size(); ++i) {
    if (!IsPrimitiveCNode(cnode->input(i), prim::kPrimDepend)) {
      continue;
    }
    auto depend_cnode = cnode->input(i)->cast<CNodePtr>();
    (void)manager->SetEdge(depend_cnode, kIndex2, pre_rely_node);
  }
}

bool MergeRecomputeCallNode(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager) {
  bool merged = false;
  std::unordered_map<FuncGraphPtr, std::map<int64_t, CNodePtr>> recompute_calls_map;
  for (const auto &node : all_nodes) {
    auto recompute_call_node = GetRecomputeCallNode(node);
    if (!recompute_call_node) {
      continue;
    }
    if (!recompute_call_node->func_graph()) {
      continue;
    }
    auto recomput_fg = GetValueNode<FuncGraphPtr>(recompute_call_node->input(kIndex0));
    if (!recomput_fg || !recomput_fg->has_attr(kRecomputeSubgraphIdAttr)) {
      continue;
    }
    auto recompute_subgraph_id = GetValue<int64_t>(recomput_fg->get_attr(kRecomputeSubgraphIdAttr));
    for (const auto &recompute_node : recomput_fg->nodes()) {
      if (!recompute_node->isa<CNode>()) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(recompute_node->cast_ptr<CNode>());
      recompute_node->cast_ptr<CNode>()->AddAttr(kRecomputeSubgraphIdAttr, MakeValue(recompute_subgraph_id));
    }
    recompute_calls_map[recompute_call_node->func_graph()][recompute_subgraph_id] = recompute_call_node;
  }

  for (const auto &recompute_call_map : recompute_calls_map) {
    size_t call_index = 0;
    AnfNodePtr pre_rely_node = nullptr;
    for (const auto &call_pair : recompute_call_map.second) {
      // Find all inputs control edge rely node
      if (call_index % kSizeTwo == 0) {
        pre_rely_node = GetDependRelyNode(call_pair.second);
        ++call_index;
        continue;
      }
      if (pre_rely_node) {
        ReplaceDependRelyNode(call_pair.second, pre_rely_node);
        merged = true;
      }
      ++call_index;
    }
  }
  return merged;
}
}  // namespace

bool MergeRecomputeCallNodes(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  if ((parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
       parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel)) {
    return false;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->CellReuseLevel() == CellReuseLevel::kNoCellReuse) {
    return false;
  }
  auto recompute_comm_overlap = ms_context->get_param<std::string>(MS_CTX_RECOMPUTE_COMM_OVERLAP);
  if (recompute_comm_overlap.empty()) {
    return false;
  }
  if (root->has_flag(MERGE_RECOMPUTE_CALL_NODES_RUN_ONCE_ONLY)) {
    return false;
  }
  FuncGraphManagerPtr manager;
  pipeline::ResourceBasePtr res;
  if (optimizer == nullptr) {
    manager = root->manager();
    res = std::make_shared<pipeline::Resource>();
    res->set_manager(manager);
  } else {
    res = optimizer->resource();
    MS_EXCEPTION_IF_NULL(res);
    manager = res->manager();
  }
  MS_EXCEPTION_IF_NULL(manager);
  CNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  const auto &all_nodes = TopoSort(ret, SuccDeeperSimple);
  bool changed = MergeRecomputeCallNode(all_nodes, manager);
  if (changed) {
    root->set_flag(MERGE_RECOMPUTE_CALL_NODES_RUN_ONCE_ONLY, true);
  }
  DumpGraph(root, std::string("merge_recompute_call_nodes"));
  return changed;
}
}  // namespace parallel
}  // namespace mindspore
