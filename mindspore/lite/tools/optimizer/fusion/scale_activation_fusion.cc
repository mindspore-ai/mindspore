/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/scale_activation_fusion.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "ops/fusion/scale_fusion.h"
#include "ops/op_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
const BaseRef ScaleActivationFusion::DefinePattern() const {
  auto is_scale = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimScaleFusion>);
  MS_CHECK_TRUE_RET(is_scale != nullptr, {});
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  return VectorRef({is_activation, is_scale});
}

const AnfNodePtr ScaleActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto act_node = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(act_node != nullptr, nullptr);
  if (!CheckPrimitiveType(act_node, prim::kPrimActivation) || IsMarkedTrainOp(act_node)) {
    return nullptr;
  }
  MS_CHECK_TRUE_RET(act_node->size() == kInputSizeTwo, nullptr);
  auto act_prim = ops::GetOperator<mindspore::ops::Activation>(act_node->input(FIRST_INPUT));
  MS_CHECK_TRUE_RET(act_prim != nullptr, nullptr);
  auto act_prim_c = act_prim->GetPrim();
  MS_CHECK_TRUE_RET(act_prim_c != nullptr && act_prim_c->GetAttr(ops::kActivationType) != nullptr, nullptr);
  if (act_prim->get_activation_type() != mindspore::RELU && act_prim->get_activation_type() != mindspore::RELU6) {
    return nullptr;
  }

  auto scale_node = act_node->input(SECOND_INPUT);
  MS_CHECK_TRUE_RET(scale_node != nullptr, nullptr);
  auto scale_cnode = scale_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(scale_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(scale_cnode) || IsMultiOutputTensors(func_graph, scale_cnode)) {
    return nullptr;
  }
  auto scale_prim = ops::GetOperator<ops::ScaleFusion>(scale_cnode->input(FIRST_INPUT));
  MS_ASSERT(scale_prim != nullptr);
  auto scale_prim_c = scale_prim->GetPrim();
  MS_CHECK_TRUE_RET(scale_prim_c != nullptr, nullptr);
  ActivationType act_type = act_prim->get_activation_type();
  if (scale_prim_c->GetAttr(ops::kActivationType) != nullptr && scale_prim->get_activation_type() != NO_ACTIVATION) {
    auto scale_act = scale_prim->get_activation_type();
    MS_CHECK_TRUE_RET(scale_act == RELU || scale_act == RELU6, nullptr);
    act_type = scale_act == RELU6 ? RELU6 : act_type;
  }
  (void)scale_prim_c->AddAttr(ops::kActivationType, MakeValue<int64_t>(static_cast<int64_t>(act_type)));
  return scale_node;
}
}  // namespace mindspore::opt
