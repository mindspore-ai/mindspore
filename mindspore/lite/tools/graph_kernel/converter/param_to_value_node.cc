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
#include "tools/graph_kernel/converter/param_to_value_node.h"

#include <memory>
#include <vector>
#include <algorithm>
#include "tools/graph_kernel/common/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/other_ops.h"
#include "include/common/utils/anfalgo.h"
#include "ir/func_graph_cloner.h"
#include "ir/anf.h"
#include "mindspore/core/ops/math_ops.h"
#include "transform/graph_ir/op_adapter_map.h"

namespace mindspore::graphkernel {

bool ParamToValueNode::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  auto nodes = TopoSort(func_graph->output());
  auto manager = Manage(func_graph);
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : nodes) {
    if (node->isa<CNode>()) {
      auto cnode = std::static_pointer_cast<CNode>(node);
      if (opt::CheckPrimitiveType(node, prim::kPrimStridedSliceV2)) {
        constexpr size_t kStridedSliceConstInputBegin = 2;
        for (size_t i = kStridedSliceConstInputBegin; i < cnode->size(); i++) {
          auto input = cnode->input(i);
          if (input->isa<Parameter>()) {
            auto begin = std::dynamic_pointer_cast<Parameter>(input);
            if (begin->has_default()) {
              auto new_input = std::make_shared<ValueNode>(begin->default_param());
              new_input->set_abstract(begin->abstract());
              func_graph->AddValueNode(new_input);
              cnode->set_input(i, new_input);
            }
          }
        }
      } else if (opt::CheckPrimitiveType(node, prim::kPrimReshape)) {
        constexpr const int kReshapeTensorToParamIndex = 2;
        for (size_t i = kReshapeTensorToParamIndex; i < cnode->size(); i++) {
          auto input = cnode->input(i);
          if (input->isa<Parameter>()) {
            auto begin = std::dynamic_pointer_cast<Parameter>(input);
            if (begin->has_default()) {
              auto new_input = std::make_shared<ValueNode>(begin->default_param());
              new_input->set_abstract(begin->abstract());
              func_graph->AddValueNode(new_input);
              cnode->set_input(i, new_input);
              changed = true;
            }
          }
        }
      }
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
