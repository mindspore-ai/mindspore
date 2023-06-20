/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/cle_pattern.h"
#include <vector>
#include <unordered_map>
#include <set>
#include <memory>
#include <string>
#include "mindspore/core/ops/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/fusion/conv2d_fusion.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::lite::quant {
std::unordered_map<std::string, VectorRef> CLEPattern::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kConvWithConvPatternName] = DefineConvWithConvPattern();
  patterns[kConvWithDepthWithConvPatternName] = DefineConvWithDepthWithConvPattern();
  return patterns;
}

bool IsLinearActivation(const api::SharedPtr<ops::Conv2DFusion> &conv2d) {
  std::set<ActivationType> liner_activations = {RELU, NO_ACTIVATION};
  auto value_ptr = conv2d->GetAttr(ops::kActivationType);
  if (value_ptr == nullptr || liner_activations.find(conv2d->get_activation_type()) != liner_activations.end()) {
    return true;
  }
  return false;
}

bool IsConvNode(const BaseRef &n, ConvNode node) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    if (!opt::CheckPrimitiveType(anf_node, prim::kPrimConv2DFusion)) {
      return false;
    }
    api::SharedPtr<ops::Conv2DFusion> conv = nullptr;
    if (utils::isa<CNodePtr>(anf_node)) {
      auto c_node = anf_node->cast<CNodePtr>();
      conv = ops::GetOperator<ops::Conv2DFusion>(c_node->input(0));
    } else if (utils::isa<ValueNodePtr>(anf_node)) {
      conv = ops::GetOperator<ops::Conv2DFusion>(anf_node);
    }
    if (conv == nullptr) {
      return false;
    }
    if (!IsLinearActivation(conv)) {
      return false;
    }
    if (node == COMMON_CONV) {
      return conv->get_group() == 1;
    } else if (node == DEPTHWISE_CONV) {
      return conv->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(conv->GetAttr(ops::kIsDepthWise));
    } else {
      MS_LOG(ERROR) << "Not supported conv node type.";
      return false;
    }
  }
  return false;
}

bool IsCommonConvNode(const BaseRef &n) { return IsConvNode(n, COMMON_CONV); }

bool IsDepthWiseConvNode(const BaseRef &n) { return IsConvNode(n, DEPTHWISE_CONV); }

VectorRef CLEPattern::DefineConvWithConvPattern() const {
  auto is_conv1 = std::make_shared<CondVar>(IsCommonConvNode);
  MS_CHECK_TRUE_RET(is_conv1 != nullptr, {});
  auto is_conv2 = std::make_shared<CondVar>(IsCommonConvNode);
  MS_CHECK_TRUE_RET(is_conv2 != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_conv2, is_conv1, is_seq_var});
}

VectorRef CLEPattern::DefineConvWithDepthWithConvPattern() const {
  auto is_conv3 = std::make_shared<CondVar>(IsCommonConvNode);
  MS_CHECK_TRUE_RET(is_conv3 != nullptr, {});
  auto is_depthwise_conv2 = std::make_shared<CondVar>(IsDepthWiseConvNode);
  MS_CHECK_TRUE_RET(is_depthwise_conv2 != nullptr, {});
  auto is_conv1 = std::make_shared<CondVar>(IsCommonConvNode);
  MS_CHECK_TRUE_RET(is_conv1 != nullptr, {});
  auto is_seq_var_1 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var_1 != nullptr, {});
  auto depth_conv_ref = VectorRef({is_depthwise_conv2, is_conv1, is_seq_var_1});
  auto is_seq_var_2 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var_2 != nullptr, {});
  return VectorRef({is_conv3, depth_conv_ref, is_seq_var_2});
}

VectorRef CLEPattern::DefineDepthWithConvPattern() const {
  auto is_dw_conv1 = std::make_shared<CondVar>(IsDepthWiseConvNode);
  MS_CHECK_TRUE_RET(is_dw_conv1 != nullptr, {});
  auto is_conv2 = std::make_shared<CondVar>(IsCommonConvNode);
  MS_CHECK_TRUE_RET(is_conv2 != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_conv2, is_dw_conv1, is_seq_var});
}

AnfNodePtr CLEPattern::Process(const string &pattern_name, const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                               const EquivPtr &) const {
  if (pattern_name == kConvWithConvPatternName || pattern_name == kDepthWithConvPatternName) {
    CombinationLayer conv_conv;
    conv_conv.layer2 = node->cast<CNodePtr>();
    conv_conv.layer1 = conv_conv.layer2->input(1)->cast<CNodePtr>();
    conv_conv.layer_num = kInputsNum2;
    if (opt::IsMultiOutputTensors(func_graph, conv_conv.layer1)) {
      return nullptr;
    }
    combination_layer_.emplace_back(conv_conv);
    MS_LOG(DEBUG) << pattern_name;
    MS_LOG(DEBUG) << conv_conv.layer1->fullname_with_scope();
    MS_LOG(DEBUG) << conv_conv.layer2->fullname_with_scope();
  } else if (pattern_name == kConvWithDepthWithConvPatternName) {
    CombinationLayer conv_depth_conv;
    conv_depth_conv.layer3 = node->cast<CNodePtr>();
    conv_depth_conv.layer2 = conv_depth_conv.layer3->input(1)->cast<CNodePtr>();
    conv_depth_conv.layer1 = conv_depth_conv.layer2->input(1)->cast<CNodePtr>();
    conv_depth_conv.layer_num = kInputsNum3;
    if (opt::IsMultiOutputTensors(func_graph, conv_depth_conv.layer1) ||
        opt::IsMultiOutputTensors(func_graph, conv_depth_conv.layer2)) {
      return nullptr;
    }
    combination_layer_.emplace_back(conv_depth_conv);
    MS_LOG(DEBUG) << pattern_name;
    MS_LOG(DEBUG) << conv_depth_conv.layer1->fullname_with_scope();
    MS_LOG(DEBUG) << conv_depth_conv.layer2->fullname_with_scope();
    MS_LOG(DEBUG) << conv_depth_conv.layer3->fullname_with_scope();
  }
  return nullptr;
}
}  // namespace mindspore::lite::quant
