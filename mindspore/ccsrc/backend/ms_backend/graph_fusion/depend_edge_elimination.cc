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
#include "backend/ms_backend/graph_fusion/depend_edge_elimination.h"
#include <map>
#include <vector>
#include <memory>
#include "primitive/array_ops.h"
#include "primitive/sequence_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "ir/graph_utils.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/core/eliminate_redundant_output.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_helper.h"

namespace mindspore::graphkernel {
bool DependEdgeElimination::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto nodes = TopoSort(func_graph->get_return());
  for (auto iter = nodes.crbegin(); iter != nodes.crend(); ++iter) {
    auto node = *iter;
    if (!AnfUtils::IsGraphKernel(node)) {
      continue;
    }
    auto sub_graph = GetCNodeFuncGraph(node);
    MS_EXCEPTION_IF_NULL(sub_graph);
    auto graph_nodes = TopoSort(sub_graph->get_return());
    if (!std::any_of(graph_nodes.begin(), graph_nodes.end(), [](const AnfNodePtr &node) {
          return IsPrimitiveCNode(node, prim::kPrimBatchMatMul) || IsPrimitiveCNode(node, prim::kPrimMatMul) ||
                 IsPrimitiveCNode(node, prim::kPrimGroupedMatmul);
        })) {
      continue;
    }
    auto output = sub_graph->output();
    if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
      continue;
    }
    std::vector<AnfNodePtr> depend_nodes;
    AnfNodePtr com_node{nullptr};
    const auto &users = mng->node_users()[node];
    for (const auto &index_pair : users) {
      auto tuple_get_node = index_pair.first;
      if (IsPrimitiveCNode(tuple_get_node, prim::kPrimTupleGetItem)) {
        const auto &tuple_get_users = mng->node_users()[tuple_get_node];
        auto tuple_get_cnode = tuple_get_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(tuple_get_cnode);
        auto tuple_index = common::AnfAlgo::GetTupleGetItemOutIndex(tuple_get_cnode);
        auto make_tuple_node = output->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(make_tuple_node);
        auto subgraph_ouput = make_tuple_node->input(tuple_index + 1);
        MS_EXCEPTION_IF_NULL(subgraph_ouput);
        auto &[real_user_node, index] = tuple_get_users.back();
        if (!IsPrimitiveCNode(subgraph_ouput, prim::kPrimAssign) && tuple_get_users.size() == 1 &&
            IsPrimitiveCNode(real_user_node, prim::kPrimDepend) && index == kIndex2) {
          depend_nodes.emplace_back(real_user_node);
          continue;
        }
        com_node = tuple_get_node;
      }
    }
    if (!depend_nodes.empty() && com_node != nullptr) {
      for (auto depend_node : depend_nodes) {
        auto src_node = depend_node->cast<CNodePtr>()->input(kIndex2);
        mng->Replace(src_node, com_node);
      }
      changed = true;
    }
  }
  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
    (void)std::make_shared<EliminateHangingOutput>()->Run(func_graph);
  }
  return changed;
}
}  // namespace mindspore::graphkernel
