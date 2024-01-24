/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/merge_comm.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
namespace {
bool IsSameTargetShape(const CNodePtr &reshape_node_a, const CNodePtr &reshape_node_b) {
  auto value_ptr_a = reshape_node_a->input(kIndex2)->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  auto value_ptr_b = reshape_node_b->input(kIndex2)->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  if (value_ptr_a.size() != value_ptr_b.size()) {
    return false;
  }
  for (size_t i = 0; i < value_ptr_a.size(); i++) {
    int64_t cur_shape_a = static_cast<int64_t>(GetValue<int64_t>(value_ptr_a.at(i)));
    int64_t cur_shape_b = static_cast<int64_t>(GetValue<int64_t>(value_ptr_b.at(i)));
    if (cur_shape_a != cur_shape_b) {
      return false;
    }
  }
  return true;
}

void MergeAllGather(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager) {
  std::unordered_map<CNodePtr, std::vector<CNodePtr>> allgather_input_map;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimAllGather)) {
      continue;
    }
    auto allgather_cnode = node->cast<CNodePtr>();
    auto pre_node = GetInputNodeWithFilter(allgather_cnode->input(kIndex1), [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimReshape);
      return std::make_pair(filter, 1);
    });
    if (!IsPrimitiveCNode(pre_node)) {
      continue;
    }
    auto pre_cnode = pre_node->cast<CNodePtr>();
    allgather_input_map[pre_cnode].push_back(allgather_cnode);
  }
  for (const auto &allgather_pairs : allgather_input_map) {
    if (allgather_pairs.second.size() <= 1) {
      continue;
    }
    auto allgather_list = allgather_pairs.second;
    auto allgather_cnode1 = allgather_list.front();
    auto is_same_allgather =
      std::all_of(allgather_list.begin(), allgather_list.end(), [&allgather_cnode1](const CNodePtr &allgather_cnode2) {
        auto ag1_prim = GetCNodePrimitive(allgather_cnode1);
        auto ag2_prim = GetCNodePrimitive(allgather_cnode2);
        auto group1 = ag1_prim->GetAttr(GROUP);
        auto group2 = ag2_prim->GetAttr(GROUP);
        if (!group1 || !group2) {
          return false;
        }
        if (GetValue<std::string>(group1) != GetValue<std::string>(group2)) {
          return false;
        }
        if (IsPrimitiveCNode(allgather_cnode1->input(kIndex1), prim::kPrimReshape) !=
            IsPrimitiveCNode(allgather_cnode2->input(kIndex1), prim::kPrimReshape)) {
          return false;
        }
        if (IsPrimitiveCNode(allgather_cnode1->input(kIndex1), prim::kPrimReshape) &&
            IsPrimitiveCNode(allgather_cnode2->input(kIndex1), prim::kPrimReshape)) {
          if (!IsSameTargetShape(allgather_cnode1->input(kIndex1)->cast<CNodePtr>(),
                                 allgather_cnode2->input(kIndex1)->cast<CNodePtr>())) {
            return false;
          }
        }
        if (allgather_cnode1->func_graph() != allgather_cnode2->func_graph()) {
          return false;
        }
        return true;
      });
    if (!is_same_allgather) {
      MS_LOG(INFO) << "allgather nodes share the same input node:" << allgather_pairs.first->DebugString()
                   << " is not equal.";
      continue;
    }
    auto ag0 = allgather_list.front();
    for (const auto &ag : allgather_list) {
      manager->Replace(ag, ag0);
    }
  }
}
}  // namespace

bool MergeComm(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  auto graph_set = ForwardGraph(root);
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!IsAutoParallelCareGraph(root) || (root->has_flag(MERGE_COMM_RUN_ONCE_ONLY)) || graph_set.size() < 1) {
    return changes;
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
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  MergeAllGather(all_nodes, manager);
  DumpGraph(root, std::string("merge_comm"));

  // allreduce fusion only run once
  root->set_flag(MERGE_COMM_RUN_ONCE_ONLY, true);
  res->SetResult(pipeline::kStepParallelGraph, root);
  return changes;
}
}  // namespace parallel
}  // namespace mindspore
