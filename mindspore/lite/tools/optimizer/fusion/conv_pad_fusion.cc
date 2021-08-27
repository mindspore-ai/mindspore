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

#include "tools/optimizer/fusion/conv_pad_fusion.h"
#include <memory>
#include <vector>
#include "tools/common/tensor_util.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kPadInputsLength = 3;
constexpr size_t kConvInputIndex = 1;
constexpr size_t kConvNoBiasLen = 3;
constexpr size_t kConvWithBiasLen = 4;
constexpr size_t kFilterDimsSize = 2;
constexpr size_t NHWCTopPadPos = 2;
constexpr size_t NCHWTopPadPos = 4;
constexpr size_t kTop = 0;
constexpr size_t kBottom = 1;
constexpr size_t kLeft = 2;
constexpr size_t kRight = 3;
constexpr size_t kPadDims = 4;

void ReplaceParamsAndNodes(const FuncGraphPtr &func_graph, const CNodePtr &conv_cnode, const CNodePtr &pad_cnode,
                           const std::string &pattern_name) {
  auto paddings = pad_cnode->input(kInputIndexTwo)->cast<ParameterPtr>();
  auto pad_list = std::dynamic_pointer_cast<tensor::Tensor>(paddings->default_param());
  auto pad_data = static_cast<int32_t *>(pad_list->data_c());

  std::vector<int64_t> pad_list_data;
  if (pattern_name == "PadConvPatternName") {
    pad_list_data.push_back(pad_data[kTop + NHWCTopPadPos]);
    pad_list_data.push_back(pad_data[kBottom + NHWCTopPadPos]);
    pad_list_data.push_back(pad_data[kLeft + NHWCTopPadPos]);
    pad_list_data.push_back(pad_data[kRight + NHWCTopPadPos]);
  } else {
    pad_list_data.push_back(pad_data[kTop + NCHWTopPadPos]);
    pad_list_data.push_back(pad_data[kBottom + NCHWTopPadPos]);
    pad_list_data.push_back(pad_data[kLeft + NCHWTopPadPos]);
    pad_list_data.push_back(pad_data[kRight + NCHWTopPadPos]);
  }

  auto conv_primitive = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(conv_cnode->input(0));
  MS_ASSERT(conv_primitive != nullptr);
  int64_t conv_pad_mode = conv_primitive->GetAttr(ops::kPadMode) == nullptr ? 0 : conv_primitive->get_pad_mode();
  if (conv_pad_mode == PadMode::PAD) {
    auto pad_list_node = conv_primitive->GetAttr(ops::kPadList);
    if (pad_list_node != nullptr) {
      std::vector<int64_t> conv_pad_list = GetValue<std::vector<int64_t>>(pad_list_node);
      if (conv_pad_list.size() == kPadDims) {
        pad_list_data[kTop] += conv_pad_list[kTop];
        pad_list_data[kBottom] += conv_pad_list[kBottom];
        pad_list_data[kLeft] += conv_pad_list[kLeft];
        pad_list_data[kRight] += conv_pad_list[kRight];
      }
    }
  } else if (conv_pad_mode == PadMode::SAME) {
    ValuePtr kernel_node = conv_primitive->GetAttr(ops::kKernelSize);
    MS_ASSERT(kernel_node != nullptr);
    std::vector<int64_t> kernel_list = GetValue<std::vector<int64_t>>(kernel_node);
    if (kernel_list.size() != kFilterDimsSize) {
      MS_LOG(ERROR) << "Filter Dims should be 2, Fusion failed! ,name:" << conv_cnode->fullname_with_scope();
      return;
    } else if (kernel_list[0] == kernel_list[1]) {
      int64_t pad_size = std::floor(kernel_list[0] / 2);
      for (size_t i = 0; i < pad_list_data.size(); ++i) {
        pad_list_data[i] += pad_size;
      }
    } else {
      int64_t top_pad_size = std::floor(kernel_list[0] / 2);
      int64_t left_pad_size = std::floor(kernel_list[1] / 2);
      pad_list_data[kTop] += top_pad_size;
      pad_list_data[kBottom] += top_pad_size;
      pad_list_data[kLeft] += left_pad_size;
      pad_list_data[kRight] += left_pad_size;
    }
    conv_primitive->set_pad_mode(PadMode::PAD);
  } else {
    conv_primitive->set_pad_mode(PadMode::PAD);
  }
  conv_primitive->set_pad_list(pad_list_data);

  // delete padFusion
  auto manager = func_graph->manager();
  manager->Replace(pad_cnode, pad_cnode->input(1));
}

bool IsPrimitiveProper(const CNodePtr &conv_cnode, const CNodePtr &pad_cnode) {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(pad_cnode != nullptr);
  if (!utils::isa<Parameter>(pad_cnode->input(kInputIndexTwo))) {
    return false;
  }
  auto pad_primitive = GetValueNode<std::shared_ptr<ops::PadFusion>>(pad_cnode->input(0));
  MS_ASSERT(pad_primitive != nullptr);
  int64_t pad_mode = pad_primitive->get_padding_mode();
  if (pad_mode != PaddingMode::CONSTANT) {
    return false;
  }
  ValuePtr pad_constant_node = pad_primitive->GetAttr(ops::kConstantValue);
  MS_ASSERT(pad_constant_node != nullptr);
  float pad_value = GetValue<float>(pad_constant_node);
  if (pad_value != 0) {
    return false;
  }

  return true;
}
}  // namespace

VectorRef ConvPadFusion::DefinePadConvPattern() const {
  auto pad_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPadFusion>);
  auto conv_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimConv2DFusion>);
  auto weight_var = std::make_shared<CondVar>(IsParamNode);
  auto bias_var = std::make_shared<SeqVar>();
  return VectorRef({conv_var, pad_var, weight_var, bias_var});
}

VectorRef ConvPadFusion::DefinePadTransposeConvPattern() const {
  auto pad_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPadFusion>);
  auto transpose_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  VectorRef transpose_conv_ref = VectorRef({transpose_var, pad_var, transpose_param});

  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto weight_var = std::make_shared<CondVar>(IsParamNode);
  auto bias_var = std::make_shared<SeqVar>();
  VectorRef trans_conv_ref = VectorRef({conv_var, transpose_conv_ref, weight_var, bias_var});
  return trans_conv_ref;
}

std::unordered_map<std::string, VectorRef> ConvPadFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["PadConvPatternName"] = DefinePadConvPattern();
  patterns["PadTransposeConvPatternName"] = DefinePadTransposeConvPattern();
  return patterns;
}

AnfNodePtr ConvPadFusion::Process(const std::string &pattern_name, const FuncGraphPtr &func_graph,
                                  const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  auto conv_cnode = node->cast<CNodePtr>();
  MS_ASSERT(conv_cnode != nullptr);
  if (conv_cnode->inputs().size() != kConvWithBiasLen && conv_cnode->inputs().size() != kConvNoBiasLen) {
    MS_LOG(WARNING) << "conv node inputs error ,name:" << conv_cnode->fullname_with_scope();
    return nullptr;
  }
  CNodePtr pad_cnode = nullptr;
  if (pattern_name == "PadTransposeConvPatternName") {
    CNodePtr transpose_cnode = conv_cnode->input(1)->cast<CNodePtr>();
    if (IsMultiOutputTensors(func_graph, transpose_cnode)) {
      MS_LOG(WARNING) << "transpose node is used as input by multiple cnodes, Fusion failed! ,name:"
                      << transpose_cnode->fullname_with_scope();
      return nullptr;
    }
    MS_ASSERT(transpose_cnode != nullptr);
    pad_cnode = transpose_cnode->input(1)->cast<CNodePtr>();
  } else {
    pad_cnode = conv_cnode->input(1)->cast<CNodePtr>();
  }

  if (IsMultiOutputTensors(func_graph, pad_cnode)) {
    MS_LOG(WARNING) << "pad node is used as input by multiple cnodes, Fusion failed! ,name:"
                    << pad_cnode->fullname_with_scope();
    return nullptr;
  }

  MS_ASSERT(pad_cnode != nullptr);
  if (pad_cnode == nullptr || pad_cnode->size() != kInputSizeThree) {
    MS_LOG(WARNING) << "pad node inputs error ,name:" << pad_cnode->fullname_with_scope();
    return nullptr;
  }

  if (!IsPrimitiveProper(conv_cnode, pad_cnode)) {
    MS_LOG(WARNING) << conv_cnode->fullname_with_scope() << " is not match with previous "
                    << pad_cnode->fullname_with_scope() << " op. Fusion failed!";
    return nullptr;
  }

  ReplaceParamsAndNodes(func_graph, conv_cnode, pad_cnode, pattern_name);
  return nullptr;
}

}  // namespace opt
}  // namespace mindspore
