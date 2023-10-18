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
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"
#include "tools/optimizer/graph/decrease_transpose_algo.h"
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
    // auto prim_func = GetPrimitiveFunction(cnode);
    auto prim_func = GetValuePtr<Primitive>(cnode->input(0));
    if (prim_func == nullptr) {
      // cnode is attr primitive node, do nothing
      continue;
    }

    auto node_inputs = cnode->inputs();
    std::vector<AnfNodePtr> new_node_inputs;

    // change PrimtiveFunction into Primitive
    auto op_type = prim_func->name();
    auto prim = CreatePrimitive(op_type);
    if (prim == nullptr) {
      MS_LOG(ERROR) << "create primitive failed";
      return false;
    }
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

    // create a new CNode and replace the old one
    // static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
    // auto op_it = operator_fns.find(op_type);
    // if (op_it == operator_fns.end()) {
    //  MS_LOG(WARNING) << "unsupported op operator type: " << op_type;
    //  return false;
    //}
    // auto base_operator = op_it->second(prim);
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

PrimitiveFunctionPtr ArgsToAttrPass::GetPrimitiveFunction(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);

  auto prim_input = cnode->input(0);
  if (!IsValueNode<Primitive>(prim_input)) {
    return nullptr;
  }
  auto prim = GetValuePtr<Primitive>(prim_input);
  if (!prim->isa<PrimitiveFunction>()) {
    return nullptr;
  }
  return prim->cast<PrimitiveFunctionPtr>();
}

PrimitivePtr ArgsToAttrPass::CreatePrimitive(const std::string &op_type) {
#if 0
  static auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  std::shared_ptr<mindspore::Primitive> prim;
  auto it = op_primc_fns.find(op_type);
  if (it == op_primc_fns.end()) {
    MS_LOG(WARNING) << "unsupported op primitive type: " << op_type;
    return nullptr;
  }
  prim = it->second();
  prim->set_instance_name(op_type);
  return prim;
#else
  auto base_operator = std::make_shared<ops::BaseOperator>(op_type);
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "create base operator failed";
    return nullptr;
  }
  return base_operator->GetPrim();
#endif
}
// bool ArgsToAttrPass::Test(const std::string &op_type) {
//   static auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
//   std::shared_ptr<mindspore::Primitive> prim;
//   auto it = op_primc_fns.find(op_type);
//   if (it == op_primc_fns.end()) {
//     MS_LOG(WARNING) << "MindirModelLoader: Convert primitives failed, unsupported op primitive type: " << op_type;
//     continue;
//   }
//   prim = it->second();
//   prim->set_instance_name(op_type);
//   // for (int j = 0; j < primitive_proto.attribute_size(); j++) {
//   //   auto attr_proto = primitive_proto.attribute(j);
//   //   auto value_ptr = MindirModelUtil::MakeValueFromAttribute(attr_proto);
//   //   MS_CHECK_TRUE_MSG(value_ptr != nullptr, false,
//   //                     "MindirModelLoader: convert primitives failed, parse prim: "
//   //                       << prim->ToString() << " attributes error: " << attr_proto.DebugString());
//   //   (void)prim->AddAttr(attr_proto.name(), value_ptr);
//   // }
//   // static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
//   // auto op_it = operator_fns.find(op_type);
//   // if (op_it == operator_fns.end()) {
//   //   MS_LOG(WARNING) << "MindirModelLoader: Convert primitives failed, unsupported op operator type: " << op_type;
//   //   continue;
//   // }
//   // auto base_operator = op_it->second(prim);
//   // MS_CHECK_TRUE_MSG(this->all_operators_.count(primitive_proto.name()) <= 0, false,
//   //                   "MindirModelLoader: There is a duplication primitive instance name: " <<
//   primitive_proto.name());
//   // this->all_operators_[primitive_proto.name()] = base_operator;
//   return true;
// }
}  // namespace opt
}  // namespace mindspore
