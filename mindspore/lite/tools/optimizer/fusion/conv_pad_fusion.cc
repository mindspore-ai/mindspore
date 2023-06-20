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
#include "tools/optimizer/fusion/conv_pad_fusion.h"
#include <memory>
#include <vector>
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "tools/common/tensor_util.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "src/common/utils.h"

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
constexpr int kPadElementNum = 8;

void ReplaceParamsAndNodes(const FuncGraphPtr &func_graph, const CNodePtr &conv_cnode, const CNodePtr &pad_cnode,
                           const std::string &pattern_name) {
  auto paddings = pad_cnode->input(kInputIndexTwo)->cast<ParameterPtr>();
  MS_ASSERT(paddings != nullptr);
  MS_ASSERT(paddings->default_param() != nullptr);
  auto pad_list = std::dynamic_pointer_cast<tensor::Tensor>(paddings->default_param());
  MS_ASSERT(pad_list != nullptr);
  MS_ASSERT(pad_list->ElementsNum() == kPadElementNum);
  auto pad_data = static_cast<int32_t *>(pad_list->data_c());
  MS_ASSERT(pad_data != nullptr);

  std::vector<int64_t> pad_list_data;
  std::vector<int> zero_pos;
  if (pattern_name == "PadConvPatternName") {
    zero_pos = {0, 1, kLeft + NCHWTopPadPos, kRight + NCHWTopPadPos};
    pad_list_data.push_back(pad_data[kTop + NHWCTopPadPos]);
    pad_list_data.push_back(pad_data[kBottom + NHWCTopPadPos]);
    pad_list_data.push_back(pad_data[kLeft + NHWCTopPadPos]);
    pad_list_data.push_back(pad_data[kRight + NHWCTopPadPos]);
  } else {
    zero_pos = {0, 1, kTop + NHWCTopPadPos, kBottom + NHWCTopPadPos};
    pad_list_data.push_back(pad_data[kTop + NCHWTopPadPos]);
    pad_list_data.push_back(pad_data[kBottom + NCHWTopPadPos]);
    pad_list_data.push_back(pad_data[kLeft + NCHWTopPadPos]);
    pad_list_data.push_back(pad_data[kRight + NCHWTopPadPos]);
  }
  if (std::any_of(zero_pos.begin(), zero_pos.end(), [&pad_data](int pos) { return pad_data[pos] != 0; })) {
    return;
  }

  auto conv_primitive = ops::GetOperator<ops::Conv2DFusion>(conv_cnode->input(0));
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
    auto kernel_node = conv_primitive->GetAttr(ops::kKernelSize);
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
  MS_ASSERT(manager != nullptr);
  (void)manager->Replace(pad_cnode, pad_cnode->input(1));
}

bool IsPrimitiveProper(const CNodePtr &pad_cnode) {
  MS_ASSERT(pad_cnode != nullptr);
  if (!utils::isa<Parameter>(pad_cnode->input(kInputIndexTwo))) {
    return false;
  }
  auto pad_list = pad_cnode->input(kInputIndexTwo)->cast<ParameterPtr>();
  auto tensor_param = pad_list->default_param();
  if (tensor_param == nullptr) {
    return false;
  }
  auto tensor = tensor_param->cast<tensor::TensorPtr>();
  if (tensor == nullptr) {
    return false;
  }
  if (tensor->data_type() != kNumberTypeInt32 && tensor->data_type() != kNumberTypeInt) {
    return false;
  }
  if (tensor->data_c() == nullptr || tensor->ElementsNum() != kPadElementNum) {
    return false;
  }
  auto prim = GetValueNode<PrimitiveCPtr>(pad_cnode->input(0));
  MS_ASSERT(prim != nullptr);
  auto pad_primitive = api::MakeShared<ops::PadFusion>(prim);
  if (!prim->HasAttr(ops::kPaddingMode)) {
    return false;
  }
  MS_ASSERT(pad_primitive != nullptr);
  int64_t pad_mode = pad_primitive->get_padding_mode();
  if (pad_mode != PaddingMode::CONSTANT) {
    return false;
  }
  float pad_value = 0;
  if (pad_cnode->size() == kInputSizeThree) {
    auto pad_constant_node = pad_primitive->GetAttr(ops::kConstantValue);
    if (pad_constant_node == nullptr) {
      return false;
    }
    pad_value = GetValue<float>(pad_constant_node);
  } else {
    MS_ASSERT(pad_cnode->size() == kInputSizeFour);
    if (pad_cnode->input(kInputIndexThree)->isa<CNode>()) {
      return false;
    }
    lite::DataInfo data_info;
    auto ret = lite::FetchConstData(pad_cnode, kInputIndexThree, converter::kFmkTypeMs, &data_info, false);
    MS_CHECK_TRUE_RET(ret == lite::RET_OK, lite::RET_ERROR);
    if (data_info.data_ptr_ == nullptr) {
      return false;
    }
    if (data_info.data_type_ != kNumberTypeFloat32 && data_info.data_type_ != kNumberTypeFloat) {
      return false;
    }
    pad_value = *static_cast<float *>(data_info.data_ptr_);
  }
  if (!mindspore::lite::FloatCompare(pad_value)) {
    return false;
  }

  return true;
}
}  // namespace

VectorRef ConvPadFusion::DefinePadConvPattern() const {
  auto is_pad = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPadFusion>);
  MS_CHECK_TRUE_RET(is_pad != nullptr, {});
  auto is_conv = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimConv2DFusion>);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_conv, is_pad, is_param, is_seq_var});
}

VectorRef ConvPadFusion::DefinePadTransposeConvPattern() const {
  auto is_pad = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPadFusion>);
  MS_CHECK_TRUE_RET(is_pad != nullptr, {});
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_param_perm = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_perm != nullptr, {});
  VectorRef transpose_conv_ref = VectorRef({is_transpose, is_pad, is_param_perm});

  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto is_param_weight = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_weight != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  VectorRef trans_conv_ref = VectorRef({is_conv, transpose_conv_ref, is_param_weight, is_seq_var});
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
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  auto conv_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(conv_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(conv_cnode)) {
    return nullptr;
  }
  if (conv_cnode->inputs().size() != kConvWithBiasLen && conv_cnode->inputs().size() != kConvNoBiasLen) {
    MS_LOG(INFO) << "conv node inputs error ,name:" << conv_cnode->fullname_with_scope();
    return nullptr;
  }
  CNodePtr pad_cnode = nullptr;
  if (pattern_name == "PadTransposeConvPatternName") {
    auto transpose_cnode = conv_cnode->input(1)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(transpose_cnode != nullptr, nullptr);
    if (IsMarkedTrainOp(transpose_cnode)) {
      return nullptr;
    }
    if (IsMultiOutputTensors(func_graph, transpose_cnode)) {
      MS_LOG(INFO) << "transpose node is used as input by multiple cnodes, Fusion failed! ,name:"
                   << transpose_cnode->fullname_with_scope();
      return nullptr;
    }
    MS_ASSERT(transpose_cnode != nullptr);
    pad_cnode = transpose_cnode->input(1)->cast<CNodePtr>();
  } else {
    pad_cnode = conv_cnode->input(1)->cast<CNodePtr>();
    if (IsMarkedTrainOp(pad_cnode)) {
      return nullptr;
    }
  }
  MS_CHECK_TRUE_RET(pad_cnode != nullptr, nullptr);

  if (IsMultiOutputTensors(func_graph, pad_cnode)) {
    MS_LOG(INFO) << "pad node is used as input by multiple cnodes, Fusion failed! ,name:"
                 << pad_cnode->fullname_with_scope();
    return nullptr;
  }

  if (pad_cnode->size() != kInputSizeThree && pad_cnode->size() != kInputSizeFour) {
    MS_LOG(INFO) << "pad node inputs error ,name:" << pad_cnode->fullname_with_scope();
    return nullptr;
  }

  if (!IsPrimitiveProper(pad_cnode)) {
    MS_LOG(INFO) << conv_cnode->fullname_with_scope() << " does not match with previous "
                 << pad_cnode->fullname_with_scope() << " op. Fusion failed!";
    return nullptr;
  }

  ReplaceParamsAndNodes(func_graph, conv_cnode, pad_cnode, pattern_name);
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
