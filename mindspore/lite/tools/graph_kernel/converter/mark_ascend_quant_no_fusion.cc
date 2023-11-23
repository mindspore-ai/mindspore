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
#include "tools/graph_kernel/converter/mark_ascend_quant_no_fusion.h"

#include <memory>
#include <vector>
#include <algorithm>
#include "tools/graph_kernel/common/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "backend/common/graph_kernel/adapter/expander.h"
#include "ops/other_ops.h"
#include "include/common/utils/anfalgo.h"
#include "ir/func_graph_cloner.h"
#include "ir/anf.h"
#include "mindspore/core/ops/math_ops.h"
#include "transform/graph_ir/op_adapter_map.h"

namespace mindspore::graphkernel {

constexpr auto kKeepBasic = "keep_basic";
constexpr size_t kMaxNoFusionDepth = 2;
constexpr size_t kStartDepth = 0;

void SetNoFusionRecursive(const AnfNodePtr &anf_node, int current_depth, int max_depth,
                          std::function<std::vector<AnfNodePtr>(const CNodePtr &)> child_nodes_func) {
  if (current_depth > max_depth || !anf_node->isa<CNode>()) {
    return;
  }
  auto cnode = std::dynamic_pointer_cast<CNode>(anf_node);
  MS_EXCEPTION_IF_NULL(cnode);
  cnode->AddAttr(kKeepBasic, MakeValue(true));
  for (auto &child_node : child_nodes_func(cnode)) {
    SetNoFusionRecursive(child_node, current_depth + 1, max_depth, child_nodes_func);
  }
}

bool MarkAscendQuantNoFusion::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  auto nodes = TopoSort(func_graph->output());
  auto ascend_quant = std::make_shared<Primitive>(transform::kNameAscendQuant);
  auto manager = Manage(func_graph);
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : nodes) {
    if (node->isa<CNode>()) {
      if (opt::CheckPrimitiveType(node, prim::kPrimNPUAntiQuant)) {
        SetNoFusionRecursive(node, kStartDepth, kMaxNoFusionDepth, [&manager](const CNodePtr &cnode) {
          auto cnode_outputs = manager->node_users()[cnode];
          std::vector<AnfNodePtr> ret;
          (void)std::transform(cnode_outputs.begin(), cnode_outputs.end(), std::back_inserter(ret),
                               [](const auto &index_set) { return index_set.first; });
          return ret;
        });
        changed = true;
      } else if (opt::CheckPrimitiveType(node, ascend_quant)) {
        SetNoFusionRecursive(node, kStartDepth, kMaxNoFusionDepth,
                             [](const CNodePtr &cnode) { return cnode->inputs(); });
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
