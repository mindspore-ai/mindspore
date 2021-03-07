/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/fusion/conv_tuple_activation_fusion.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kActivationInputsLength = 2;
bool IsTupleGetItemNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem);
  }
  return false;
}
}  // namespace
const BaseRef ConvTupleActivationFusion::DefinePattern() const {
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto tuple_getitem_var = std::make_shared<CondVar>(IsTupleGetItemNode);
  auto tuple_index = std::make_shared<Var>();
  VectorRef tuple_get_item = VectorRef({tuple_getitem_var, conv_var, tuple_index});
  auto act_var = std::make_shared<CondVar>(IsActivationNode);
  return VectorRef({act_var, tuple_get_item});
}

const AnfNodePtr ConvTupleActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
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
  MS_ASSERT(act_prim != nullptr);
  if (act_prim->GetAttr(ops::kActivationType) == nullptr ||
      (act_prim->get_activation_type() != mindspore::RELU && act_prim->get_activation_type() != mindspore::RELU6)) {
    return nullptr;
  }
  AnfNodePtr tuple_node = act_node->input(1);
  MS_ASSERT(tuple_node != nullptr);
  auto tuple_cnode = tuple_node->cast<CNodePtr>();
  auto conv_node = tuple_cnode->input(1);
  if (CheckIfAnfNodeIsNull(conv_node) != lite::RET_OK) {
    return nullptr;
  }
  if (conv_node != nullptr && conv_node->isa<CNode>()) {
    if (IsMultiOutputTensors(func_graph, conv_node)) {
      return nullptr;
    }
    auto conv_cnode = conv_node->cast<CNodePtr>();
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
