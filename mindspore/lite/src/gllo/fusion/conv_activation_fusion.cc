/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *conv_activation_fusion.h
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/gllo/fusion/conv_activation_fusion.h"
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/ir/primitive_t_value.h"
#include "mindspore/ccsrc/utils/utils.h"
#include "src/gllo/common/utils.h"

namespace mindspore {
namespace opt {
const BaseRef ConvActivationFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  // conv2d inputs may be 2 or 3 inputs,match move to process
  auto prim = new schema::PrimitiveT();
  prim->value.type = primitive_type;
  auto prim_value = std::make_shared<lite::PrimitiveTValue>(prim);

  return VectorRef({prim_value, X});
}

const AnfNodePtr ConvActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_LOG(DEBUG) << "conv activation pass process:" << schema::EnumNamesPrimitiveType()[primitive_type];
  CheckIfFuncGraphIsNull(func_graph);

  CheckIfAnfNodeIsNull(node);
  auto act_node = node->cast<CNodePtr>();
  CheckIfCNodeIsNull(act_node);
  CheckInputSize(act_node, 2);

  auto act_primitive = GetValueNode<std::shared_ptr<lite::PrimitiveTValue>>(act_node->input(0));
  if (act_primitive->GetPrimitiveT()->value.AsActivation()->type != activation_type) {
    return node;
  }
  AnfNodePtr pre_node = act_node->input(1);
  CheckIfAnfNodeIsNull(pre_node);
  if (pre_node != nullptr && pre_node->isa<CNode>()) {
    auto conv_node = pre_node->cast<CNodePtr>();
    auto node_type = GetCNodeType(conv_node);
    if (node_type == schema::PrimitiveType_Conv2D || node_type == schema::PrimitiveType_DepthwiseConv2D) {
      auto primitiveT_value = GetValueNode<std::shared_ptr<lite::PrimitiveTValue>>(conv_node->input(0));
      primitiveT_value->GetPrimitiveT()->value.AsConv2D()->activationType = activation_type;
      return pre_node;
    }
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
