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

#include "tools/optimizer/graph/send_op_add_control_depend.h"

#include <vector>
#include "ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
const auto kDataToControl = "data_to_control";
}
const BaseRef SendOpAddControlDepend::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  return VectorRef({prim::kPrimSend, x1});
}

#if defined(_WIN32) || defined(_WIN64)
const AnfNodePtr SendOpAddControlDepend::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  return node;
}
#else
const AnfNodePtr SendOpAddControlDepend::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsPrimitiveCNode(cnode, prim::kPrimSend)) {
    return nullptr;
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive != nullptr && primitive->HasAttr(kDataToControl)) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "Process node: " << node->fullname_with_scope()
                << ", input node: " << cnode->input(1)->fullname_with_scope();

  auto tensor = std::make_shared<tensor::Tensor>(0.0);

  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto value = MakeValue(tensor);
  MS_CHECK_TRUE_RET(value != nullptr, nullptr);
  auto value_node = std::make_shared<ValueNode>(value);
  value_node->set_abstract(tensor->ToAbstract());
  MS_EXCEPTION_IF_NULL(value_node);
  func_graph->AddValueNode(value_node);

  std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(kDependOpName)), value_node, cnode};
  auto depend_node = NewCNode(depend_input, func_graph);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_fullname_with_scope(node->fullname_with_scope() + "_Depend");
  depend_node->set_scope(node->scope());
  depend_node->set_abstract(value_node->abstract());
  MS_LOG(INFO) << "Create new depend: " << depend_node->fullname_with_scope()
               << " for node: " << cnode->fullname_with_scope();
  common::AnfAlgo::SetNodeAttr(kDataToControl, MakeValue(true), cnode);
  return depend_node;
}
#endif
}  // namespace opt
}  // namespace mindspore
