/**
 * Copyright 201+ Huawei Technologies Co., Ltd
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

#include "tools/converter/legacy_optimizer/node/weight_format_pass.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"

namespace mindspore {
namespace lite {
int WeightFormatPass::Run(GraphNode *graphNode) {
  MS_ASSERT(graphNode != nullptr);
  auto status = ShapeFormatTrans(graphNode);
  if (status != 0) {
    MS_LOG(ERROR) << "ShapeFormatTrans failed: " << status;
    return status;
  }
  if (this->quantType == QuantType_AwareTraining || this->quantType == QuantType_PostTraining) {
    status = QuantDataFormatTrans(graphNode);
    if (status != 0) {
      MS_LOG(ERROR) << "QuantDataFormatTrans failed: " << status;
      return status;
    }
  } else {
    status = NonQuantDataFormatTrans(graphNode);
    if (status != 0) {
      MS_LOG(ERROR) << "NonQuantDataFormatTrans failed: " << status;
      return status;
    }
  }
  return 0;
}

void WeightFormatPass::SetQuantType(QuantType quantType) { this->quantType = quantType; }

void WeightFormatPass::SetFmkType(converter::FmkType fmkType) { this->fmkType = fmkType; }

// pre set tensor format
// non quant, filterFormat:
//           conv     deconv     depth     dedepth
// caffe   K(C/g)HW  C(K/g)HW      /         /
// tf        HWCK     HWKC       HWCK      HWKC
// onnx    K(C/g)HW  C(K/g)HW      /         /

// awareing quant, filterFormat:
//           conv     deconv     depth     dedepth
// onnx      KHWC       ?        CHWK         ?
// tf        HWCK       ?        HWCK         ?
int WeightFormatPass::ShapeFormatTrans(GraphNode *graphNode) {
  MS_ASSERT(graphNode != nullptr);
  auto &subGraph = graphNode->subGraph;
  auto &node = graphNode->opDef;
  MS_ASSERT(subGraph != nullptr);
  MS_ASSERT(node != nullptr);
  auto opType = node->primitive->value.type;
  if (opType != schema::PrimitiveType_Conv2D && opType != schema::PrimitiveType_DepthwiseConv2D &&
      opType != schema::PrimitiveType_DeConv2D && opType != schema::PrimitiveType_DeDepthwiseConv2D) {
    return 0;
  }
  MS_ASSERT(node->inputIndex.size() >= 2);
  auto weightIndex = node->inputIndex.at(1);
  MS_ASSERT(subGraph->allTensors.size() > weightIndex);
  auto &weightTensor = subGraph->allTensors[weightIndex];
  auto &shape = weightTensor->dims;
  MS_ASSERT(shape.size() == 4);
  if (fmkType == converter::FmkType_CAFFE) {
    switch (node->quantType) {
      case QuantType_QUANT_NONE: {
        if (opType == schema::PrimitiveType_Conv2D || opType == schema::PrimitiveType_DepthwiseConv2D ||
            opType == schema::PrimitiveType_DeConv2D || opType == schema::PrimitiveType_DeDepthwiseConv2D) {
          weightTensor->format = schema::Format_KCHW;
        } else {
          MS_LOG(ERROR) << "Invalid opType: " << schema::EnumNamePrimitiveType(opType)
                        << ", node: " << node->name.c_str();
          return -1;
        }
      } break;
      default: {
        MS_LOG(ERROR) << "Invalid quantType: " << schema::EnumNameQuantType(node->quantType)
                      << ", node: " << node->name.c_str();
        return -1;
      }
    }
    return 0;
  } else if (fmkType == converter::FmkType_MS) {
    switch (node->quantType) {
      case QuantType_AwareTraining: {
        if (opType == schema::PrimitiveType_Conv2D || opType == schema::PrimitiveType_DepthwiseConv2D) {
          weightTensor->format = schema::Format_HWCK;
        } else {
          weightTensor->format = schema::Format_HWKC;
        }
      } break;
      case QuantType_QUANT_NONE: {
        // conv [filter_height, filter_width, in_channels, out_channels]
        // depthwise [filter_height, filter_width, in_channels, channel_multiplier]
        if (opType == schema::PrimitiveType_Conv2D) {
          weightTensor->format = schema::Format_KCHW;
        } else if (opType == schema::PrimitiveType_DepthwiseConv2D) {
          weightTensor->format = schema::Format_KCHW;
        } else {
          MS_LOG(ERROR) << "Unsupported opType: " << EnumNamePrimitiveType(opType) << ", node: " << node->name;
          return -1;
        }
      } break;
      default: {
        MS_LOG(ERROR) << "Invalid opType: %d, node: " << opType, node->name.c_str();
        return -1;
      }
    }
    return 0;
  } else if (fmkType == converter::FmkType_TF) {
    switch (node->quantType) {
      case QuantType_AwareTraining: {
        if (opType == schema::PrimitiveType_Conv2D || opType == schema::PrimitiveType_DepthwiseConv2D) {
          weightTensor->format = schema::Format_HWCK;
        } else {
          weightTensor->format = schema::Format_HWKC;
        }
      } break;
      case QuantType_QUANT_NONE: {
        // conv [filter_height, filter_width, in_channels, out_channels]
        // depthwise [filter_height, filter_width, in_channels, channel_multiplier]
        if (opType == schema::PrimitiveType_Conv2D || opType == schema::PrimitiveType_DepthwiseConv2D) {
          weightTensor->format = schema::Format_HWCK;
        } else {
          weightTensor->format = schema::Format_HWKC;
        }
      } break;
      default: {
        MS_LOG(ERROR) << "Invalid opType: %d, node: " << opType, node->name.c_str();
        return -1;
      }
    }
    return 0;
  } else if (fmkType == converter::FmkType_TFLITE) {
    switch (node->quantType) {
      case QuantType_QUANT_NONE:
      case QuantType_AwareTraining:
      case QuantType_PostTraining: {
        if (opType == schema::PrimitiveType_Conv2D) {
          weightTensor->format = schema::Format_KHWC;
        } else if (opType == schema::PrimitiveType_DepthwiseConv2D) {
          weightTensor->format = schema::Format_CHWK;
        } else if (opType == schema::PrimitiveType_DeConv2D) {
          weightTensor->format = schema::Format_CHWK;
        } else {
          MS_LOG(ERROR) << "Unsupported format";
          return -1;
        }
      } break;
      default: {
        MS_LOG(ERROR) << "Invalid opType: %d, node: " << opType, node->name.c_str();
        return -1;
      }
    }
    MS_LOG(DEBUG) << "weight_tensor_format: " << weightTensor->format;
    return 0;
  } else if (fmkType == converter::FmkType_ONNX) {
    switch (node->quantType) {
      case QuantType_AwareTraining: {
        // sum up from current onnx quant models
        if (opType == schema::PrimitiveType_Conv2D) {
          weightTensor->format = schema::Format_KHWC;
        } else if (opType == schema::PrimitiveType_DepthwiseConv2D) {
          weightTensor->format = schema::Format_CHWK;
        } else {
          MS_LOG(ERROR) << "Invalid opType: %d, node: " << opType, node->name.c_str();
          return -1;
        }
      } break;
      case QuantType_QUANT_NONE: {
        // conv (K x C/group x kH x kW) group = 1
        // depth (K x C/group x kH x kW) group = channelOut ==> (K, multiplier, H, W)
        // deconv (C x K/group x kH x kW) group = 1
        // dedepth (C x K/group x kH x kW) group = channelIn ==> (C, multiplier, H, W)
        if (opType == schema::PrimitiveType_Conv2D) {
          weightTensor->format = schema::Format_KCHW;
        } else if (opType == schema::PrimitiveType_DepthwiseConv2D) {
          weightTensor->format = schema::Format_KCHW;
        } else if (opType == schema::PrimitiveType_DeConv2D) {
          weightTensor->format = schema::Format_CKHW;
        } else {
          MS_LOG(ERROR) << "Invalid opType: %d, node: " << opType, node->name.c_str();
          return -1;
        }
      } break;
      default: {
        MS_LOG(ERROR) << "Unsupported quantType: %d, node: " << node->quantType, node->name.c_str();
        return -1;
      }
    }
  } else {
    MS_LOG(ERROR) << "Invalid fmkType: %d, node: " << fmkType, node->name.c_str();
    return -1;
  }
  return 0;
}

// inference needed filterFormat:
//           conv     deconv     depth     dedepth
// uint8     KHWC     KHWC       KHWC      KHWC
int WeightFormatPass::QuantDataFormatTrans(GraphNode *graphNode) {
  MS_ASSERT(graphNode != nullptr);
  auto &subGraph = graphNode->subGraph;
  auto &node = graphNode->opDef;
  MS_ASSERT(subGraph != nullptr);
  MS_ASSERT(node != nullptr);
  auto opType = node->primitive->value.type;
  if (opType != schema::PrimitiveType_Conv2D && opType != schema::PrimitiveType_DepthwiseConv2D &&
      opType != schema::PrimitiveType_DeConv2D && opType != schema::PrimitiveType_DeDepthwiseConv2D) {
    return RET_OK;
  }

  MS_ASSERT(node->inputIndex.size() >= 2);
  auto weightIndex = node->inputIndex.at(1);
  MS_ASSERT(subGraph->allTensors.size() > weightIndex);
  auto &weightTensor = subGraph->allTensors[weightIndex];
  MS_ASSERT(weightTensor->dataType == kNumberTypeInt8);  // DataType_DT_FLOAT
  STATUS status = RET_OK;
  if (opType == schema::PrimitiveType_Conv2D) {         // weight should be KHWC
    if (weightTensor->format == schema::Format_KCHW) {  // from caffe
      if (weightTensor->dataType == kNumberTypeInt8) {  // DataType_DT_UINT8) {
        MS_LOG(DEBUG) << "**weight tensor index: %d, format: %d, datatype: " << weightIndex << weightTensor->format
                      << weightTensor->dataType;
        status = TransFilterFormat<int8_t>(weightTensor.get(), kKCHW2HWCK);
      } else {
        MS_LOG(DEBUG) << "--weight tensor index: %d, format: %d, datatype: " << weightIndex << weightTensor->format
                      << weightTensor->dataType;
        status = TransFilterFormat<float>(weightTensor.get(), kKCHW2KHWC);
      }
    } else if (weightTensor->format != schema::Format_KHWC) {
      MS_LOG(ERROR) << "Unsupported weightTensor format: " << weightTensor->format;
      return -1;
    }
    if (status == 0) {
      node->primitive->value.AsConv2D()->format = schema::Format_NHWC;
      weightTensor->format = schema::Format_KHWC;
    } else {
      MS_LOG(WARNING) << "TransFilter %sToKHWC failed, node : "
                      << (weightTensor->format == schema::Format_KHWC ? "KHWC" : "KCHW") << node->name.c_str();
      // todo(00445839): consider varible weight condition
    }
  } else if (opType == schema::PrimitiveType_DepthwiseConv2D) {  // weight should be KHWC
    if (weightTensor->format == schema::Format_CKHW) {           // from caffe
      if (weightTensor->dataType == kNumberTypeInt8) {           // DataType_DT_UINT8) {
        MS_LOG(DEBUG) << "**weight tensor index: " << weightIndex << "format: " << weightTensor->format
                      << "datatype: " << weightTensor->dataType;
        status = TransFilterFormat<int8_t>(weightTensor.get(), kCKHW2KHWC);
      } else if (weightTensor->dataType == kNumberTypeUInt8) {
        MS_LOG(DEBUG) << "**weight tensor index: " << weightIndex << "format: " << weightTensor->format
                      << "datatype: " << weightTensor->dataType;
        status = TransFilterFormat<uint8_t>(weightTensor.get(), kCKHW2KHWC);
      } else {
        MS_LOG(DEBUG) << "**weight tensor index: " << weightIndex << "format: " << weightTensor->format
                      << "datatype: " << weightTensor->dataType;
        status = TransFilterFormat<float>(weightTensor.get(), kCKHW2KHWC);
      }

    } else if (weightTensor->format == schema::Format_CHWK) {  // from onnx
      if (weightTensor->dataType == kNumberTypeInt8) {
        MS_LOG(DEBUG) << "**weight tensor index: " << weightIndex << "format: " << weightTensor->format
                      << "datatype: " << weightTensor->dataType;
        status = TransFilterFormat<int8_t>(weightTensor.get(), kCHWK2KHWC);
      } else if (weightTensor->dataType == kNumberTypeUInt8) {
        MS_LOG(DEBUG) << "**weight tensor index: " << weightIndex << "format: " << weightTensor->format
                      << "datatype: " << weightTensor->dataType;
        status = TransFilterFormat<uint8_t>(weightTensor.get(), kCHWK2KHWC);
      } else {
        MS_LOG(DEBUG) << "**weight tensor index: " << weightIndex << "format: " << weightTensor->format
                      << "datatype: " << weightTensor->dataType;
        status = TransFilterFormat<float>(weightTensor.get(), kCHWK2KHWC);
      }
    } else if (weightTensor->format != schema::Format_KHWC) {
      MS_LOG(ERROR) << "Unsupported weightTensor format: " << weightTensor->format;
      return -1;
    }
    if (status == 0) {
      node->primitive->value.AsDepthwiseConv2D()->format = schema::Format_NHWC;
      weightTensor->format = schema::Format_KHWC;
    } else {
      MS_LOG(WARNING) << "TransFilter" << (weightTensor->format == schema::Format_KHWC ? "KHWC" : "CKHW")
                      << "To KHWC failed, node : " << node->name.c_str();
      // todo(00445839): consider varible weight condition
    }
  } else {  // weight should be HWCK
    node->primitive->value.AsDeConv2D()->format = schema::Format_NHWC;
    weightTensor->format = schema::Format_KHWC;
  }
  return 0;
}

// inference needed filterFormat:
//           conv     deconv     depth     dedepth
// fp32      KCHW     CKHW       CKHW      CKHW
int WeightFormatPass::NonQuantDataFormatTrans(GraphNode *graphNode) {
  MS_ASSERT(graphNode != nullptr);
  auto &subGraph = graphNode->subGraph;
  auto &node = graphNode->opDef;
  MS_ASSERT(subGraph != nullptr);
  MS_ASSERT(node != nullptr);
  auto opType = node->primitive->value.type;
  if (opType != schema::PrimitiveType_Conv2D && opType != schema::PrimitiveType_DepthwiseConv2D &&
      opType != schema::PrimitiveType_DeConv2D && opType != schema::PrimitiveType_DeDepthwiseConv2D) {
    return 0;
  }

  MS_ASSERT(node->inputIndex.size() >= 2);
  auto weightIndex = node->inputIndex.at(1);
  MS_ASSERT(subGraph->allTensors.size() > weightIndex);
  auto &weightTensor = subGraph->allTensors[weightIndex];
  if (weightTensor->dataType != TypeId::kNumberTypeFloat32) {
    MS_LOG(ERROR) << "weight tensor data should be float";
    //    return -1;
  }
  STATUS status = RET_OK;
  if (opType == schema::PrimitiveType_Conv2D) {         // weight should be KCHW
    if (weightTensor->format == schema::Format_KCHW) {  // from caffe or onnx or ms
      status = TransFilterFormat<float>(weightTensor.get(), kKCHW2KHWC);
    } else if (weightTensor->format == schema::Format_KHWC) {
      status = RET_OK;
    } else if (weightTensor->format == schema::Format_CHWK) {
      status = TransFilterFormat<float>(weightTensor.get(), kCHWK2KHWC);
    } else {
      MS_LOG(ERROR) << "Unsupported weightTensor format: " << weightTensor->format;
      return -1;
    }
    if (status == 0) {
      node->primitive->value.AsConv2D()->format = schema::Format_NHWC;
      weightTensor->format = schema::Format_KHWC;
    } else {
      MS_LOG(WARNING) << "TransFilter " << ((weightTensor->format == schema::Format_HWCK) ? "HWCK" : "NHWC")
                      << "ToKCHW failed, node : " << node->name.c_str();
      // todo(00445839): consider varible weight condition
    }
  } else if (opType == schema::PrimitiveType_DepthwiseConv2D) {  // weight should be CKHW
    if (fmkType == converter::FmkType_MS) {
      weightTensor->format = schema::Format_CKHW;
    }
    if (weightTensor->format == schema::Format_CKHW) {  // from caffe or onnx or ms
      status = TransFilterFormat<float>(weightTensor.get(), kCKHW2KHWC);
    } else if (weightTensor->format == schema::Format_KCHW) {
      status = TransFilterFormat<float>(weightTensor.get(), kKCHW2KHWC);
    } else if (weightTensor->format == schema::Format_CHWK) {
      status = TransFilterFormat<float>(weightTensor.get(), kCHWK2KHWC);
    } else {
      MS_LOG(ERROR) << "Unsupported weightTensor format: " << weightTensor->format;
      return -1;
    }
    if (status == 0) {
      node->primitive->value.AsDepthwiseConv2D()->format = schema::Format_NHWC;
      weightTensor->format = schema::Format_KHWC;
    } else {
      MS_LOG(WARNING) << "TransFilter HWCKToCKHW failed, node : " << node->name.c_str();
      // todo(00445839): consider varible weight condition
    }
  } else if (opType == schema::PrimitiveType_DeConv2D) {  // weight should be KHWC
    if (weightTensor->format == schema::Format_KCHW) {    // from caffe or onnx or ms
      status = TransFilterFormat<float>(weightTensor.get(), kKCHW2KHWC);
    } else if (weightTensor->format == schema::Format_KHWC) {  // from tf
      status = RET_OK;
    } else {
      MS_LOG(ERROR) << "Unsupported weightTensor format: " << weightTensor->format;
      return -1;
    }
    if (status == 0) {
      node->primitive->value.AsDeConv2D()->format = schema::Format_NCHW;
      weightTensor->format = schema::Format_KHWC;
    } else {
      MS_LOG(WARNING) << "TransFilter HWKCToKCHW failed, node : " << node->name.c_str();
      // todo(00445839): consider varible weight condition
    }
  } else if (opType == schema::PrimitiveType_DeDepthwiseConv2D) {  // weight should be KHWC
    if (weightTensor->format == schema::Format_KHWC) {
      return 0;
    } else if (weightTensor->format == schema::Format_KCHW) {  // from caffe
      status = TransFilterFormat<float>(weightTensor.get(), kKCHW2KHWC);
    } else if (weightTensor->format == schema::Format_HWKC) {  // from tf or onnx
      status = TransFilterFormat<float>(weightTensor.get(), kHWKC2CKHW);
    } else {
      MS_LOG(ERROR) << "Unsupported weightTensor format: " << weightTensor->format;
      return -1;
    }
    if (status == 0) {
      node->primitive->value.AsDeDepthwiseConv2D()->format = schema::Format_NHWC;
      weightTensor->format = schema::Format_CKHW;
    } else {
      MS_LOG(WARNING) << "TransFilter HWKCToCKHW failed, node : " << node->name.c_str();
      // todo(00445839): consider varible weight condition
    }
  }
  return 0;
}
}  // namespace lite
}  // namespace mindspore
