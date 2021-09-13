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

#include "tools/optimizer/fusion/conv_tuple_activation_fusion.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
const BaseRef ConvTupleActivationFusion::DefinePattern() const {
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto is_tuple_getitem = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTupleGetItem>);
  MS_CHECK_TRUE_RET(is_tuple_getitem != nullptr, {});
  auto is_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var != nullptr, {});
  auto tuple_get_item = VectorRef({is_tuple_getitem, is_conv, is_var});
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  return VectorRef({is_activation, tuple_get_item});
}

const AnfNodePtr ConvTupleActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto act_node = node->cast<CNodePtr>();
  if (act_node == nullptr || act_node->size() != kInputSizeTwo) {
    return nullptr;
  }
  if (IsMarkedTrainOp(act_node)) {
    return nullptr;
  }
  if (!CheckPrimitiveType(act_node, prim::kPrimActivation)) {
    return nullptr;
  }
  auto act_prim = GetValueNode<std::shared_ptr<mindspore::ops::Activation>>(act_node->input(0));
  MS_ASSERT(act_prim != nullptr);
  if (act_prim->GetAttr(ops::kActivationType) == nullptr ||
      (act_prim->get_activation_type() != mindspore::RELU && act_prim->get_activation_type() != mindspore::RELU6)) {
    return nullptr;
  }
  AnfNodePtr tuple_node = act_node->input(1);
  MS_ASSERT(tuple_node != nullptr);
  auto tuple_cnode = tuple_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(tuple_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(tuple_cnode)) {
    return nullptr;
  }
  auto conv_node = tuple_cnode->input(1);
  if (conv_node != nullptr && conv_node->isa<CNode>()) {
    auto conv_cnode = conv_node->cast<CNodePtr>();
    if (IsMarkedTrainOp(conv_cnode)) {
      return nullptr;
    }
    if (IsMultiOutputTensors(func_graph, conv_node)) {
      return nullptr;
    }
    if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
      auto primc = GetValueNode<std::shared_ptr<mindspore::ops::Conv2DFusion>>(conv_cnode->input(0));
      MS_ASSERT(primc != nullptr);
      if (primc->GetAttr(ops::kActivationType) == nullptr || primc->get_activation_type() == mindspore::NO_ACTIVATION) {
        primc->set_activation_type(act_prim->get_activation_type());
        conv_node->set_abstract(act_node->abstract());
        return conv_node;
      }
    } else {
      MS_LOG(ERROR) << "conv activation pass match only conv2d or depthwise_conv2d ";
    }
  }
  return nullptr;
}
}  // namespace mindspore::opt
