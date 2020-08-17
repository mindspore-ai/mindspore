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

#include "tools/converter/legacy_optimizer/graph/weight_format_hardcode_pass.h"
#include "tools/common/converter_op_utils.h"
#include "utils/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
void WeightFormatHardCodePass::SetQuantType(QuantType quantType) { this->quantType = quantType; }

void WeightFormatHardCodePass::SetFmkType(converter::FmkType fmkType) { this->fmkType = fmkType; }

// pre set tensor format
// non quant, filterFormat:
//           conv     deconv     depth     dedepth
// caffe   K(C/g)HW  C(K/g)HW      /         /     // todo with deconvOp
// tf        HWCK     HWKC       HWCK      HWKC
// onnx    K(C/g)HW  C(K/g)HW      /         /

// awareing quant, filterFormat:
//           conv     deconv     depth     dedepth
// onnx      KHWC       ?        CHWK         ?
// tf        HWCK       ?        HWCK         ?
STATUS WeightFormatHardCodePass::Run(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto &node : graph->nodes) {
    MS_ASSERT(node != nullptr);
    MS_ASSERT(node->primitive != nullptr);
    auto opType = node->primitive->value.type;
    if (opType != PrimitiveType_Conv2D && opType != PrimitiveType_DepthwiseConv2D && opType != PrimitiveType_DeConv2D &&
        opType != PrimitiveType_DeDepthwiseConv2D) {
      continue;
    }
    MS_ASSERT(node->inputIndex.size() >= 2);
    auto weightIndex = node->inputIndex.at(1);
    MS_ASSERT(subGraph->allTensors.size() > weightIndex);
    auto &weightTensor = graph->allTensors[weightIndex];
    MS_ASSERT(weightTensor->dims.size() == 4 || weightTensor->dims.empty());  // for conv with fakqQuant before weight
    STATUS status;
    switch (fmkType) {
      case converter::FmkType_CAFFE:
        status = HardCodeCAFFE(node, weightTensor);
        break;
      case converter::FmkType_TFLITE:
        status = HardCodeTFLITE(node, weightTensor);
        break;
      case converter::FmkType_ONNX:
        status = HardCodeONNX(node, weightTensor);
        break;
      case converter::FmkType_MS:
        status = HardCodeMS(node, weightTensor);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported fmkType: " << fmkType << ", node: " << node->name;
        return RET_ERROR;
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Format hardCode faild: " << status << ", node: " << node->name;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS WeightFormatHardCodePass::HardCodeCAFFE(const std::unique_ptr<CNodeT> &node,
                                               const std::unique_ptr<TensorT> &weightTensor) {
  MS_ASSERT(node != nullptr);
  MS_ASSERT(weightTensor != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(node->primitive != nullptr);
  auto opType = node->primitive->value.type;
  switch (this->quantType) {
    case QuantType_QUANT_NONE: {
      if (opType == schema::PrimitiveType_Conv2D || opType == schema::PrimitiveType_DepthwiseConv2D ||
          opType == schema::PrimitiveType_DeConv2D || opType == schema::PrimitiveType_DeDepthwiseConv2D) {
        weightTensor->format = Format_KCHW;
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(opType) << ", node: " << node->name;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(node->quantType) << ", node: " << node->name;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS WeightFormatHardCodePass::HardCodeONNX(const std::unique_ptr<CNodeT> &node,
                                              const std::unique_ptr<TensorT> &weightTensor) {
  MS_ASSERT(node != nullptr);
  MS_ASSERT(weightTensor != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(node->primitive != nullptr);
  auto opType = node->primitive->value.type;
  switch (this->quantType) {
    case QuantType_AwareTraining: {
      // sum up from current onnx quant models
      if (opType == PrimitiveType_Conv2D) {
        weightTensor->format = Format_KHWC;
      } else if (opType == PrimitiveType_DepthwiseConv2D) {
        weightTensor->format = Format_CHWK;
      } else if (opType == PrimitiveType_DeConv2D) {
        weightTensor->format = Format_CKHW;
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(opType) << ", node: " << node->name;
        return RET_ERROR;
      }
    } break;
    case QuantType_QUANT_NONE: {
      // conv (K x C/group x kH x kW) group = 1
      // depth (K x C/group x kH x kW) group = channelOut ==> (K, multiplier, H, W)
      // deconv (C x K/group x kH x kW) group = 1
      // dedepth (C x K/group x kH x kW) group = channelIn ==> (C, multiplier, H, W)
      if (opType == PrimitiveType_Conv2D || opType == PrimitiveType_DepthwiseConv2D) {
        weightTensor->format = Format_KCHW;
      } else if (opType == PrimitiveType_DeConv2D) {
        weightTensor->format = Format_CKHW;
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(opType) << ", node: " << node->name;
        return RET_ERROR;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(node->quantType) << ", node: " << node->name;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS WeightFormatHardCodePass::HardCodeMS(const std::unique_ptr<CNodeT> &node,
                                            const std::unique_ptr<TensorT> &weightTensor) {
  MS_ASSERT(node != nullptr);
  MS_ASSERT(weightTensor != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(node->primitive != nullptr);
  auto opType = node->primitive->value.type;
  switch (this->quantType) {
    case QuantType_AwareTraining: {
      if (opType == schema::PrimitiveType_Conv2D) {
        weightTensor->format = schema::Format_HWCK;
      } else if (opType == PrimitiveType_DepthwiseConv2D) {
        weightTensor->format = Format_CKHW;
      } else {
        weightTensor->format = schema::Format_HWKC;
      }
    } break;
    case QuantType_QUANT_NONE: {
      // sum up from current ms quant models
      if (opType == PrimitiveType_Conv2D) {
        weightTensor->format = Format_KCHW;
      } else if (opType == PrimitiveType_DepthwiseConv2D) {
        weightTensor->format = Format_CKHW;
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(opType) << ", node: " << node->name;
        return RET_ERROR;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(node->quantType) << ", node: " << node->name;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS WeightFormatHardCodePass::HardCodeTFLITE(const std::unique_ptr<CNodeT> &node,
                                                const std::unique_ptr<TensorT> &weightTensor) {
  MS_ASSERT(node != nullptr);
  MS_ASSERT(weightTensor != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(node->primitive != nullptr);
  auto opType = node->primitive->value.type;
  switch (this->quantType) {
    case QuantType_AwareTraining:
    case QuantType_PostTraining:
    case QuantType_QUANT_NONE: {
      if (opType == schema::PrimitiveType_Conv2D) {
        weightTensor->format = schema::Format_KHWC;
      } else if (opType == schema::PrimitiveType_DepthwiseConv2D) {
        weightTensor->format = schema::Format_CHWK;
      } else if (opType == schema::PrimitiveType_DeConv2D) {
        weightTensor->format = schema::Format_CHWK;
      } else {
        MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(opType) << ", node: " << node->name;
        return RET_ERROR;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(opType) << ", node: " << node->name;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
