/**
* Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/switch_not_cut.h"

#include <memory>
#include <vector>
#include "ops/other_ops.h"
#include "ops/framework_ops.h"
#include "utils/ms_context.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
bool SwitchNotCut::Run(const FuncGraphPtr &func_graph) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  static const bool is_enable_ge = (context->backend_policy() == "ge");
  if (!is_enable_ge) {
    // only support ge backend
    return false;
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : all_nodes) {
    if (IsOneOfPrimitiveCNode(node, {prim::kPrimPartial, prim::kPrimSwitch})) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->AddPrimalAttr(kAttrNotCut, MakeValue(true));
    } else if (utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto primitive_input = cnode->input(kAnfPrimitiveIndex);
      if (IsPrimitiveCNode(primitive_input, prim::kPrimSwitch)) {
        cnode->AddPrimalAttr(kAttrNotCut, MakeValue(true));
      }
    }
    if (IsPrimitiveCNode(node, prim::kPrimPartial)) {
      auto cnode = node->cast<CNodePtr>();
      auto partial_graph = cnode->input(kIndex1);
      auto sub_graph = common::AnfAlgo::GetValueNodeFuncGraph(partial_graph);
      sub_graph->set_flag(kFlagSwitchInline, true);
    }
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore