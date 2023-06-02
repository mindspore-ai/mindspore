/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_EXPAND_DUMP_FLAG_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_EXPAND_DUMP_FLAG_H_

#include <set>

#include "frontend/optimizer/optimizer.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore::opt::irpass {
const PrimitiveSet dump_skipped_prim_set = {prim::kPrimReturn,       prim::kPrimDepend,      prim::kPrimMakeTuple,
                                            prim::kPrimTupleGetItem, prim::kPrimUpdateState, prim::kPrimLoad,
                                            prim::kPrimPrint,        prim::kPrimPartial};

// Expand dump flag to all of cnodes if parent graph has dump flag.
class ExpandDumpFlag {
 public:
  bool operator()(const FuncGraphPtr &, const OptimizerPtr &optimizer) const {
    MS_EXCEPTION_IF_NULL(optimizer);
    auto manager = optimizer->manager();
    MS_EXCEPTION_IF_NULL(manager);
    std::set<FuncGraphPtr> seen;
    auto graph_filter = [&seen](const FuncGraphPtr &graph) {
      if (seen.find(graph) != seen.end() ||
          (graph->has_attr(FUNC_GRAPH_FLAG_DUMP) && !graph->has_flag(FUNC_GRAPH_FLAG_DUMP))) {
        return true;
      }
      return false;
    };
    for (auto &func_graph : manager->func_graphs()) {
      if (!func_graph->has_flag(FUNC_GRAPH_FLAG_DUMP) || seen.find(func_graph) != seen.end()) {
        continue;
      }
      std::set<FuncGraphPtr> traverse_graphs;

      SuccFunc succ_func = std::bind(SuccWithFilter, graph_filter, std::placeholders::_1);
      auto nodes = TopoSort(func_graph->get_return(), succ_func);
      for (const auto &node : nodes) {
        MS_EXCEPTION_IF_NULL(node);
        auto node_graph = node->func_graph();
        if (seen.find(node_graph) != seen.end()) {
          continue;
        }
        (void)traverse_graphs.insert(node_graph);
        // If the node need be ignored or the dump flag is set by false, do not set true.
        if (!node->isa<CNode>() || IsOneOfPrimitiveCNode(node, dump_skipped_prim_set) ||
            (AnfUtils::HasDumpFlag(node) && !AnfUtils::GetDumpFlag(node))) {
          continue;
        }
        AnfUtils::SetDumpFlag(node);
      }
      for (auto graph : traverse_graphs) {
        if (graph != nullptr && graph->has_attr(FUNC_GRAPH_FLAG_DUMP)) {
          graph->erase_flag(FUNC_GRAPH_FLAG_DUMP);
        }
      }
      seen.insert(traverse_graphs.cbegin(), traverse_graphs.cend());
    }
    return false;
  }
};
}  // namespace mindspore::opt::irpass

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_EXPAND_DUMP_FLAG_H_
