/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/fusion/conv_activation_fusion.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "ops/op_utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kActivationInputsLength = 2;
}
const BaseRef ConvActivationFusion::DefinePattern() const {
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto act_var = std::make_shared<CondVar>(IsActivationNode);
  return VectorRef({act_var, conv_var});
}

const AnfNodePtr ConvActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto act_node = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(act_node) != lite::RET_OK ||
      CheckInputSize(act_node, kActivationInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  if (!CheckPrimitiveType(act_node, prim::kPrimActivation)) {
    return nullptr;
  }
  auto act_prim = GetValueNode<std::shared_ptr<mindspore::ops::Activation>>(act_node->input(0));
  if (act_prim == nullptr ||
      (act_prim->GetAttr(ops::kActivationType) != nullptr && act_prim->get_activation_type() != mindspore::RELU &&
       act_prim->get_activation_type() != mindspore::RELU6)) {
    return nullptr;
  }

  AnfNodePtr pre_node = act_node->input(1);
  if (CheckIfAnfNodeIsNull(pre_node) != lite::RET_OK) {
    return nullptr;
  }
  if (pre_node != nullptr && pre_node->isa<CNode>()) {
    if (IsMultiOutputTensors(func_graph, pre_node)) {
      return nullptr;
    }
    auto conv_node = pre_node->cast<CNodePtr>();
    MS_ASSERT(primitive_c);
    if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion) ||
        CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion)) {
      auto prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
      MS_ASSERT(prim != nullptr);
      if (prim->GetAttr(ops::kActivationType) == nullptr ||
          static_cast<mindspore::ActivationType>(GetValue<int64_t>(prim->GetAttr(ops::kActivationType))) ==
            mindspore::NO_ACTIVATION) {
        if (act_prim->get_activation_type() == mindspore::RELU) {
          prim->AddAttr(ops::kActivationType, MakeValue<int64_t>(mindspore::RELU));
        } else {
          prim->AddAttr(ops::kActivationType, MakeValue<int64_t>(mindspore::RELU6));
        }
        return pre_node;
      }
    } else {
      MS_LOG(ERROR) << "conv activation pass match only conv2d or depthwise_conv2d ";
    }
  }
  return nullptr;
}
}  // namespace mindspore::opt
