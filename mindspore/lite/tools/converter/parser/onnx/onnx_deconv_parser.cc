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

#include "tools/converter/parser/onnx/onnx_deconv_parser.h"
#include <vector>
#include <memory>
#include <algorithm>

namespace mindspore {
namespace lite {
bool OnnxDeConvParser::ParseGroupDeConvolution(const std::unique_ptr<schema::DeConv2DT> &attr,
                                               schema::PrimitiveT *primitive) {
  if (attr == nullptr || attr->group != attr->channelOut || primitive == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr";
    return false;
  }
  auto deDepthwiseConv2DParam = std::make_unique<schema::DeDepthwiseConv2DT>();
  if (deDepthwiseConv2DParam == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return false;
  }
  deDepthwiseConv2DParam->format = attr->format;
  deDepthwiseConv2DParam->channelIn = attr->channelIn;
  deDepthwiseConv2DParam->channelMultiplier = attr->channelOut / attr->channelIn;
  deDepthwiseConv2DParam->kernelW = attr->kernelW;
  deDepthwiseConv2DParam->kernelH = attr->kernelH;
  deDepthwiseConv2DParam->strideW = attr->strideW;
  deDepthwiseConv2DParam->strideH = attr->strideH;
  deDepthwiseConv2DParam->padMode = attr->padMode;
  deDepthwiseConv2DParam->padUp = attr->padUp;
  deDepthwiseConv2DParam->padDown = attr->padDown;
  deDepthwiseConv2DParam->padLeft = attr->padLeft;
  deDepthwiseConv2DParam->padRight = attr->padRight;
  deDepthwiseConv2DParam->dilateW = attr->dilateW;
  deDepthwiseConv2DParam->dilateH = attr->dilateH;
  deDepthwiseConv2DParam->activationType = attr->activationType;

  primitive->value.type = schema::PrimitiveType_DeDepthwiseConv2D;
  primitive->value.value = deDepthwiseConv2DParam.release();
  return true;
}

int OnnxDeConvParser::ParseParameters(const onnx::NodeProto &onnx_node,
                                      const std::unique_ptr<schema::DeConv2DT> &attr) {
  attr->padMode = schema::PadMode_NOTSET;
  attr->group = 1;
  attr->strideW = 1;
  attr->strideH = 1;
  attr->dilateW = 1;
  attr->dilateH = 1;

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      attr->group = static_cast<int32_t>(onnx_node_attr.i());
    } else if (onnx_node_attr.name() == "dilations") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "dilations size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      attr->dilateH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->dilateW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernels") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      attr->kernelH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->kernelW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernel_shape") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      attr->kernelH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->kernelW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "auto_pad") {
      attr->padMode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "pads") {
      if (onnx_node_attr.ints().size() != 4) {
        MS_LOG(ERROR) << "pads size " << onnx_node_attr.ints().size() << " is not 4";
        return RET_ERROR;
      }
      attr->padUp = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->padLeft = static_cast<int32_t>(onnx_node_attr.ints(1));
      attr->padDown = static_cast<int32_t>(onnx_node_attr.ints(2));
      attr->padRight = static_cast<int32_t>(onnx_node_attr.ints(3));
    } else if (onnx_node_attr.name() == "strides") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "strides size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      attr->strideH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->strideW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "order") {
      if (onnx_node_attr.s() == "NHWC") {
        attr->format = schema::Format::Format_NHWC;
      } else {
        MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s().c_str();
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "output_padding") {
      attr->outputPaddingH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->outputPaddingW = static_cast<int32_t>(onnx_node_attr.ints(1));
    }
  }
  return RET_OK;
}

lite::PrimitiveC *OnnxDeConvParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                       const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx DeConvParser";
  auto attr = std::make_unique<schema::DeConv2DT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  auto status = ParseParameters(onnx_node, attr);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Parse parameters failed.";
    return nullptr;
  }

  const auto &onnx_conv_weight = onnx_node.input(1);
  auto node_iter =
    std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                 [onnx_conv_weight](const onnx::TensorProto &proto) { return proto.name() == onnx_conv_weight; });
  if (node_iter == onnx_graph.initializer().end()) {
    MS_LOG(ERROR) << "not find node: " << onnx_conv_weight.c_str();
    return nullptr;
  }
  std::vector<int> weight_shape;
  auto size = (*node_iter).dims_size();
  weight_shape.reserve(size);
  for (int i = 0; i < size; ++i) {
    weight_shape.emplace_back((*node_iter).dims(i));
  }
  if (weight_shape.size() != 4) {
    MS_LOG(ERROR) << "weight_shape.size() should be 4, but is " << weight_shape.size();
    return nullptr;
  }
  attr->channelIn = weight_shape[0];
  attr->channelOut = weight_shape[1] * attr->group;

  attr->format = schema::Format::Format_NCHW;

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  if (attr->group != 1) {
    if (!ParseGroupDeConvolution(attr, primitive.get())) {
      MS_LOG(ERROR) << "Convert DeConvolution to DeDepthwise failed, generalized group deconv hasn't support";
      return nullptr;
    }
  } else {
    primitive->value.type = schema::PrimitiveType_DeConv2D;
    primitive->value.value = attr.release();
  }

  return PrimitiveC::Create(primitive.release());
}

OnnxNodeRegistrar g_onnxDeConvParser("ConvTranspose", new OnnxDeConvParser());
}  // namespace lite
}  // namespace mindspore
