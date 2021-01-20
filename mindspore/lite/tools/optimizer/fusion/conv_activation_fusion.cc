/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/ops/primitive_c.h"
#include "src/ops/conv2d.h"
#include "src/ops/deconv2d.h"
#include "src/ops/depthwise_conv2d.h"
#include "src/ops/activation.h"
#include "schema/inner/model_generated.h"
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
  auto primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(act_node->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Activation>>(primitivec));
  auto act_primitivec = utils::cast<std::shared_ptr<mindspore::lite::Activation>>(primitivec);
  MS_ASSERT(act_primitivec != nullptr);
  if (act_primitivec->GetType() != schema::ActivationType_RELU &&
      act_primitivec->GetType() != schema::ActivationType_RELU6) {
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
    auto node_type = GetCNodeType(conv_node);
    auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(conv_node->input(0));
    MS_ASSERT(primitive_c);
    if (node_type == schema::PrimitiveType_Conv2D) {
      MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Conv2D>>(primitive_c));
      auto primc = utils::cast<std::shared_ptr<mindspore::lite::Conv2D>>(primitive_c);
      MS_ASSERT(primc != nullptr);
      if (primc->GetActivationType() == schema::ActivationType_NO_ACTIVATION) {
        primc->SetActivationType(act_primitivec->GetType());
        return pre_node;
      }
    } else if (node_type == schema::PrimitiveType_DepthwiseConv2D) {
      MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::DepthwiseConv2D>>(primitive_c));
      auto primc = utils::cast<std::shared_ptr<mindspore::lite::DepthwiseConv2D>>(primitive_c);
      MS_ASSERT(primc != nullptr);
      if (primc->GetActivationType() == schema::ActivationType_NO_ACTIVATION) {
        primc->SetActivationType(act_primitivec->GetType());
        return pre_node;
      }
    } else if (node_type == schema::PrimitiveType_DeConv2D) {
      MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::DeConv2D>>(primitive_c));
      auto primc = utils::cast<std::shared_ptr<mindspore::lite::DeConv2D>>(primitive_c);
      MS_ASSERT(primc != nullptr);
      if (primc->GetActivationType() == schema::ActivationType_NO_ACTIVATION) {
        primc->SetActivationType(act_primitivec->GetType());
        return pre_node;
      }
    } else {
      MS_LOG(ERROR) << "conv activation pass match only conv2d or depthwise_conv2d ";
    }
  }
  return nullptr;
}
}  // namespace mindspore::opt
