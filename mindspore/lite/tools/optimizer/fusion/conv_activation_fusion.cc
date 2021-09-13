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
#include "nnacl/op_base.h"

namespace mindspore::opt {
const BaseRef ConvActivationFusion::DefinePattern() const {
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  return VectorRef({is_activation, is_conv});
}

const AnfNodePtr ConvActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto act_node = node->cast<CNodePtr>();
  if (IsMarkedTrainOp(act_node)) {
    return nullptr;
  }
  if (act_node == nullptr || act_node->size() != kInputSizeTwo ||
      !CheckPrimitiveType(act_node, prim::kPrimActivation)) {
    return nullptr;
  }
  auto act_prim = GetValueNode<std::shared_ptr<mindspore::ops::Activation>>(act_node->input(0));
  if (act_prim == nullptr ||
      (act_prim->GetAttr(ops::kActivationType) != nullptr && act_prim->get_activation_type() != mindspore::RELU &&
       act_prim->get_activation_type() != mindspore::RELU6)) {
    return nullptr;
  }

  AnfNodePtr pre_node = act_node->input(1);
  if (pre_node != nullptr && pre_node->isa<CNode>()) {
    if (IsMultiOutputTensors(func_graph, pre_node)) {
      return nullptr;
    }
    auto conv_node = pre_node->cast<CNodePtr>();
    if (IsMarkedTrainOp(conv_node)) {
      return nullptr;
    }
    if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion) ||
        CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion)) {
      auto prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
      MS_ASSERT(prim != nullptr);
      if (prim->GetAttr(ops::kActivationType) == nullptr ||
          static_cast<mindspore::ActivationType>(GetValue<int64_t>(prim->GetAttr(ops::kActivationType))) ==
            mindspore::NO_ACTIVATION) {
        auto type = act_prim->get_activation_type() == mindspore::RELU ? mindspore::RELU : mindspore::RELU6;
        prim->AddAttr(ops::kActivationType, MakeValue<int64_t>(type));
        return pre_node;
      }
    } else {
      MS_LOG(ERROR) << "conv activation pass match only conv2d or depthwise_conv2d ";
    }
  }
  return nullptr;
}
}  // namespace mindspore::opt
