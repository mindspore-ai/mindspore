/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/graph/args_to_attr_pass.h"
#include "tools/common/node_util.h"
#include "src/common/log_util.h"
#include "ops/primitive_c.h"
#include "ops/base_operator.h"
#include "ops/op_def.h"

namespace mindspore {
namespace opt {
bool ArgsToAttrPass::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }

  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "get func graph manager is nullptr";
    return false;
  }

  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      continue;
    }

    auto node_inputs = cnode->inputs();
    std::vector<AnfNodePtr> new_node_inputs;

    // change PrimtiveFunction into Primitive
    if (prim->GetAttr("primitive_function") == nullptr) {
      continue;
    }
    auto op_type = prim->name();
    auto op_def = mindspore::ops::GetOpDef(op_type);
    if (op_def == nullptr) {
      MS_LOG(DEBUG) << "cannot get op def for " << op_type;
      continue;
    }
    for (auto arg : op_def->args_) {
      auto index_it = op_def->indexes_.find(arg.arg_name_);
      if (index_it == op_def->indexes_.end()) {
        // no arg passed, skip or set default value, current skip
        continue;
      }
      auto arg_index = index_it->second;

      if (!arg.as_init_arg_) {
        // origin is input , put the node input into new node inputs vector
        new_node_inputs.emplace_back(node_inputs[arg_index + 1]);
        continue;
      }

      auto arg_input_node = cnode->input(arg_index + 1);
      if (!arg_input_node->isa<ValueNode>()) {
        // arg is not ValueNode, Network has dynamic args, not support
        MS_LOG(DEBUG) << "node " << node->fullname_with_scope() << " with arg " << arg_input_node->fullname_with_scope()
                      << " is dynamic, not support";
        continue;
      }
      auto arg_value_node = arg_input_node->cast<ValueNodePtr>();
      auto arg_value = arg_value_node->value();
      prim->AddAttr(arg.arg_name_, arg_value);
    }

    auto new_node = func_graph->NewCNode(prim, new_node_inputs);
    new_node->set_abstract(node->abstract());
    new_node->set_fullname_with_scope(node->fullname_with_scope());

    if (!manager->Replace(node, new_node)) {
      MS_LOG(ERROR) << "replace node " << node->fullname_with_scope() << " failed";
      return false;
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
