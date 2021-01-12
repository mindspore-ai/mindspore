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
#include "tools/optimizer/graph/weight_format_hardcode_pass.h"
#include <memory>
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
}  // namespace
void WeightFormatHardCodePass::SetQuantType(QuantType type) { this->quant_type = type; }
void WeightFormatHardCodePass::SetFmkType(FmkType type) { this->fmk_type = type; }
lite::STATUS WeightFormatHardCodePass::HardCodeCAFFE(const AnfNodePtr &conv_node,
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

lite::STATUS WeightFormatHardCodePass::HardCodeONNX(const AnfNodePtr &conv_node,
                                                    const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto op_type = GetCNodeType(conv_node);
  switch (this->quant_type) {
    case QuantType_AwareTraining: {
      // sum up from current onnx quant models
      if (op_type == schema::PrimitiveType_Conv2D) {
        param_value->set_format(schema::Format::Format_KHWC);
      } else if (op_type == schema::PrimitiveType_DepthwiseConv2D) {
        param_value->set_format(schema::Format::Format_CHWK);
      } else if (op_type == schema::PrimitiveType_DeConv2D) {
        param_value->set_format(schema::Format::Format_KCHW);
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(op_type)
                      << ", node: " << conv_node->fullname_with_scope();
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
      if (op_type == schema::PrimitiveType_Conv2D || op_type == schema::PrimitiveType_DepthwiseConv2D ||
          op_type == schema::PrimitiveType_DeConv2D || op_type == schema::PrimitiveType_DeDepthwiseConv2D) {
        if (param_value->format() == schema::Format::Format_NHWC) {
          param_value->set_format(schema::Format::Format_KHWC);
        } else {
          param_value->set_format(schema::Format::Format_KCHW);
        }
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(op_type)
                      << ", node: " << conv_node->fullname_with_scope();
        return lite::RET_ERROR;
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

lite::STATUS WeightFormatHardCodePass::HardCodeMS(const AnfNodePtr &conv_node,
                                                  const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto weight_node = conv_node->cast<CNodePtr>()->input(kConvWeightIndex);
  auto op_type = GetCNodeType(conv_node);
  switch (this->quant_type) {
    case QuantType_AwareTraining: {
      if (op_type == schema::PrimitiveType_Conv2D) {
        param_value->set_format(schema::Format::Format_KCHW);
      } else if (op_type == schema::PrimitiveType_DepthwiseConv2D) {
        param_value->set_format(schema::Format::Format_CKHW);
      } else {
        param_value->set_format(schema::Format::Format_KCHW);
      }
    } break;
    case QuantType_PostTraining:
    case QuantType_WeightQuant:
    case QuantType_QUANT_NONE: {
      // sum up from current ms quant models
      if (op_type == schema::PrimitiveType_Conv2D) {
        param_value->set_format(schema::Format::Format_KCHW);
      } else if (op_type == schema::PrimitiveType_DepthwiseConv2D) {
        // the format should be set to KCHW while the weight is output of constfolding .
        if (weight_node->fullname_with_scope().find("constfold") == weight_node->fullname_with_scope().npos) {
          param_value->set_format(schema::Format::Format_CKHW);
        }
      } else if (op_type == schema::PrimitiveType_DeDepthwiseConv2D) {
        param_value->set_format(schema::Format::Format_CKHW);
      } else if (op_type == schema::PrimitiveType_DeConv2D) {
        param_value->set_format(schema::Format::Format_KCHW);
#ifdef SUPPORT_TRAIN
      } else if (op_type == schema::PrimitiveType_Conv2DGradInput) {
        param_value->set_format(schema::Format::Format_KCHW);
      } else if (op_type == schema::PrimitiveType_GroupConv2DGradInput) {
        param_value->set_format(schema::Format::Format_CKHW);
#endif
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(op_type)
                      << ", node: " << conv_node->fullname_with_scope();
        return lite::RET_ERROR;
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

lite::STATUS WeightFormatHardCodePass::HardCodeTFLITE(const AnfNodePtr &conv_node,
                                                      const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto op_type = GetCNodeType(conv_node);
  switch (this->quant_type) {
    case QuantType_AwareTraining:
    case QuantType_PostTraining:
    case QuantType_WeightQuant:
    case QuantType_QUANT_NONE: {
      if (op_type == schema::PrimitiveType_Conv2D) {
        param_value->set_format(schema::Format::Format_KHWC);
      } else if (op_type == schema::PrimitiveType_DepthwiseConv2D) {
        param_value->set_format(schema::Format::Format_CHWK);
      } else if (op_type == schema::PrimitiveType_DeConv2D) {
        param_value->set_format(schema::Format::Format_CHWK);
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(op_type)
                      << ", node: " << conv_node->fullname_with_scope();
        return lite::RET_ERROR;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(op_type)
                    << ", node: " << conv_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

lite::STATUS WeightFormatHardCodePass::HardCodeTF(const AnfNodePtr &conv_node,
                                                  const ParamValueLitePtr &param_value) const {
  MS_ASSERT(conv_cnode != nullptr);
  MS_ASSERT(param_value != nullptr);
  auto op_type = GetCNodeType(conv_node);

  if (op_type == schema::PrimitiveType_Conv2D) {
    param_value->set_format(schema::Format::Format_HWCK);
  } else if (op_type == schema::PrimitiveType_DepthwiseConv2D) {
    param_value->set_format(schema::Format::Format_HWKC);
  } else {
    MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(op_type)
                  << ", node: " << conv_node->fullname_with_scope();
    return lite::RET_ERROR;
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
    auto type = opt::GetCNodeType(node);
    if (type != schema::PrimitiveType_Conv2D && type != schema::PrimitiveType_DepthwiseConv2D &&
#ifdef SUPPORT_TRAIN
        ((type != schema::PrimitiveType_Conv2DGradInput) || (fmk_type != FmkType_MS)) &&
        ((type != schema::PrimitiveType_GroupConv2DGradInput) || (fmk_type != FmkType_MS)) &&
#endif
        type != schema::PrimitiveType_DeConv2D && type != schema::PrimitiveType_DeDepthwiseConv2D) {
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
        status = HardCodeCAFFE(node, param_value);
        break;
      case FmkType_TFLITE:
        status = HardCodeTFLITE(node, param_value);
        break;
      case FmkType_TF:
        status = HardCodeTF(node, param_value);
        break;
      case FmkType_ONNX:
        status = HardCodeONNX(node, param_value);
        break;
      case FmkType_MS:
        status = HardCodeMS(node, param_value);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported fmkType: " << fmk_type << ", node: " << node->fullname_with_scope();
        return false;
    }
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "schema::Format hardCode faild: " << status << ", node: " << node->fullname_with_scope();
      return false;
    }
  }
  return false;
}
}  // namespace mindspore::opt
