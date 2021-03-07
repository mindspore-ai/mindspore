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
#include "tools/optimizer/graph/weight_format_hardcode_pass.h"
#include <memory>
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_backprop_input_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"

using mindspore::lite::converter::FmkType_CAFFE;
using mindspore::lite::converter::FmkType_MS;
using mindspore::lite::converter::FmkType_ONNX;
using mindspore::lite::converter::FmkType_TF;
using mindspore::lite::converter::FmkType_TFLITE;
using mindspore::schema::QuantType_AwareTraining;
using mindspore::schema::QuantType_PostTraining;
using mindspore::schema::QuantType_QUANT_NONE;
using mindspore::schema::QuantType_WeightQuant;
namespace mindspore::opt {
namespace {
constexpr size_t kConvWeightIndex = 2;
const PrimitivePtr kPrimConv2DBackpropInputFusion = std::make_shared<Primitive>(ops::kNameConv2DBackpropInputFusion);
}  // namespace
void WeightFormatHardCodePass::SetQuantType(QuantType type) { this->quant_type = type; }
void WeightFormatHardCodePass::SetFmkType(FmkType type) { this->fmk_type = type; }
lite::STATUS WeightFormatHardCodePass::HardCodeCAFFE(const CNodePtr &conv_node,
                                                     const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  switch (quant_type) {
    case schema::QuantType_PostTraining:
    case QuantType_WeightQuant:
    case QuantType_QUANT_NONE:
      param_value->set_format(schema::Format::Format_KCHW);
      break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(quant_type)
                    << ", node: " << conv_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

lite::STATUS WeightFormatHardCodePass::HardCodeONNX(const CNodePtr &conv_node,
                                                    const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  int64_t format = prim->GetAttr(ops::kFormat) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kFormat)) : 0;
  switch (this->quant_type) {
    case QuantType_AwareTraining: {
      // sum up from current onnx quant models
      if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
        if (!is_depth_wise) {
          param_value->set_format(schema::Format::Format_KHWC);
        } else {
          param_value->set_format(schema::Format::Format_CHWK);
        }
      } else if (CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
        param_value->set_format(schema::Format::Format_KCHW);
      } else {
        MS_LOG(ERROR) << "Unsupported op: " << conv_node->fullname_with_scope();
        return lite::RET_ERROR;
      }
    } break;
    case QuantType_PostTraining:
    case QuantType_WeightQuant:
    case QuantType_QUANT_NONE: {
      // conv (K x C/group x kH x kW) group = 1
      // depth (K x C/group x kH x kW) group = channelOut ==> (K, multiplier, H, W)
      // deconv (C x K/group x kH x kW) group = 1
      // dedepth (C x K/group x kH x kW) group = channelIn ==> (C, multiplier, H, W)
      if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion) ||
          CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion)) {
        if (format == schema::Format::Format_NHWC) {
          param_value->set_format(schema::Format::Format_KHWC);
        } else {
          param_value->set_format(schema::Format::Format_KCHW);
        }
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(quant_type)
                    << ", node: " << conv_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

lite::STATUS WeightFormatHardCodePass::HardCodeMS(const CNodePtr &conv_node,
                                                  const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  auto weight_node = conv_node->input(kConvWeightIndex);
  switch (this->quant_type) {
    case QuantType_AwareTraining:
    case QuantType_PostTraining:
    case QuantType_WeightQuant:
    case QuantType_QUANT_NONE: {
      // sum up from current ms quant models
      param_value->set_format(schema::Format::Format_KCHW);
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(quant_type)
                    << ", node: " << conv_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

lite::STATUS WeightFormatHardCodePass::HardCodeTFLITE(const CNodePtr &conv_node,
                                                      const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  switch (this->quant_type) {
    case QuantType_AwareTraining:
    case QuantType_PostTraining:
    case QuantType_WeightQuant:
    case QuantType_QUANT_NONE: {
      if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
        if (!is_depth_wise) {
          param_value->set_format(schema::Format::Format_KHWC);
        } else {
          param_value->set_format(schema::Format::Format_CHWK);
        }
      } else if (CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
        param_value->set_format(schema::Format::Format_CHWK);
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported op: " << conv_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

lite::STATUS WeightFormatHardCodePass::HardCodeTF(const CNodePtr &conv_node,
                                                  const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
    {
      if (!is_depth_wise) {
        param_value->set_format(schema::Format::Format_HWCK);
      } else {
        param_value->set_format(schema::Format::Format_HWKC);
      }
    }
  } else if (CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
    param_value->set_format(schema::Format::Format_HWCK);
  }
  return lite::RET_OK;
}

bool WeightFormatHardCodePass::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto conv_cnode = node->cast<CNodePtr>();
    if (!CheckPrimitiveType(node, prim::kPrimConv2DFusion) &&
        (!CheckPrimitiveType(node, kPrimConv2DBackpropInputFusion) || (fmk_type != FmkType_MS)) &&
        !CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
      continue;
    }
    MS_ASSERT(conv_cnode->inputs().size() > kConvWeightIndex);
    auto weight_node = conv_cnode->input(kConvWeightIndex);
    MS_ASSERT(weight_node != nullptr);
    auto param_value = GetLiteParamValue(weight_node);
    if (param_value == nullptr) {
      MS_LOG(ERROR) << "weight node must param value";
      return false;
    }
    lite::STATUS status;
    switch (fmk_type) {
      case FmkType_CAFFE:
        status = HardCodeCAFFE(conv_cnode, param_value);
        break;
      case FmkType_TFLITE:
        status = HardCodeTFLITE(conv_cnode, param_value);
        break;
      case FmkType_TF:
        status = HardCodeTF(conv_cnode, param_value);
        break;
      case FmkType_ONNX:
        status = HardCodeONNX(conv_cnode, param_value);
        break;
      case FmkType_MS:
        status = HardCodeMS(conv_cnode, param_value);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported fmkType: " << fmk_type << ", node: " << node->fullname_with_scope();
        return false;
    }
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "Format hard code failed: " << status << ", node: " << node->fullname_with_scope();
      return false;
    }
  }
  return false;
}
}  // namespace mindspore::opt
