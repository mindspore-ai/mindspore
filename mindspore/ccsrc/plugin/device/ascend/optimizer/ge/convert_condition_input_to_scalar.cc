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
#include "plugin/device/ascend/optimizer/ge/convert_condition_input_to_scalar.h"

#include <vector>
#include <memory>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ir/value.h"

namespace mindspore {
namespace opt {
constexpr auto kCondInputIndex = 1;

const BaseRef ConvertCondInputToScalar::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  return VectorRef({prim::kPrimSwitch, x1, x2, x3});
}
const AnfNodePtr ConvertCondInputToScalar::Process(const FuncGraphPtr &funcGraphPtr, const AnfNodePtr &nodePtr,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(funcGraphPtr);
  MS_EXCEPTION_IF_NULL(nodePtr);
  auto cnode = nodePtr->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto cond_input = cnode->input(kCondInputIndex);
  MS_EXCEPTION_IF_NULL(cond_input);
  auto shape = common::AnfAlgo::GetOutputInferShape(cond_input, 0);
  if (shape.empty()) {
    MS_LOG(DEBUG) << "Condition input is scalar, node: " << nodePtr->fullname_with_scope()
                  << ", cond input: " << cond_input->fullname_with_scope();
    return nodePtr;
  }
  if (shape.size() != 1 || shape[0] != 1) {
    MS_LOG(WARNING) << "Condition input is not scalar ,node:" << nodePtr->fullname_with_scope()
                    << ", cond input:" << cond_input->fullname_with_scope();
    return nodePtr;
  }

  std::vector<ValuePtr> value_list;
  value_list.emplace_back(MakeValue(0));
  auto value_tuple = std::make_shared<ValueTuple>(value_list);
  auto value_node = NewValueNode(value_tuple);
  value_node->set_abstract(value_tuple->ToAbstract());

  auto prim = std::make_shared<Primitive>(prim::kPrimReduceAny->name());
  std::vector<AnfNodePtr> reduce_any_input = {NewValueNode(prim), cond_input, value_node};
  auto reduce_any_node = funcGraphPtr->NewCNode(reduce_any_input);
  MS_EXCEPTION_IF_NULL(reduce_any_node);
  auto type_id = common::AnfAlgo::GetOutputInferDataType(cond_input, 0);
  common::AnfAlgo::SetOutputInferTypeAndShape({type_id}, {std::vector<int64_t>()}, reduce_any_node.get());
  common::AnfAlgo::SetNodeAttr(kAttrKeepDims, MakeValue(false), reduce_any_node);
  reduce_any_node->set_scope(cond_input->scope());
  common::AnfAlgo::SetNodeInput(cnode, reduce_any_node, kCondInputIndex - 1);
  MS_LOG(DEBUG) << "Insert ReduceAny for not-scalar condition input, node: " << reduce_any_node->fullname_with_scope();
  return nodePtr;
}
}  // namespace opt
}  // namespace mindspore
