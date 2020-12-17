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

#include "tools/converter/parser/onnx/onnx_conv_parser.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace mindspore::lite {
bool OnnxConvParser::ParseGroupConvolution(const std::unique_ptr<schema::Conv2DT> &attr,
                                           schema::PrimitiveT *primitive) {
  MS_LOG(DEBUG) << "onnx DepthwiseConvParser";
  if (attr == nullptr || primitive == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr";
    return false;
  }
  auto depthwiseConv2DParam = std::make_unique<schema::DepthwiseConv2DT>();
  if (depthwiseConv2DParam == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return false;
  }
  depthwiseConv2DParam->format = attr->format;
  depthwiseConv2DParam->channelIn = attr->channelIn;
  depthwiseConv2DParam->channelMultiplier = attr->channelOut / attr->channelIn;
  depthwiseConv2DParam->kernelW = attr->kernelW;
  depthwiseConv2DParam->kernelH = attr->kernelH;
  depthwiseConv2DParam->strideW = attr->strideW;
  depthwiseConv2DParam->strideH = attr->strideH;
  depthwiseConv2DParam->padMode = attr->padMode;
  depthwiseConv2DParam->padUp = attr->padUp;
  depthwiseConv2DParam->padDown = attr->padDown;
  depthwiseConv2DParam->padLeft = attr->padLeft;
  depthwiseConv2DParam->padRight = attr->padRight;
  depthwiseConv2DParam->dilateW = attr->dilateW;
  depthwiseConv2DParam->dilateH = attr->dilateH;
  depthwiseConv2DParam->activationType = attr->activationType;

  primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  primitive->value.value = depthwiseConv2DParam.release();
  return true;
}

lite::PrimitiveC *OnnxConvParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx ConvParser";
  auto attr = std::make_unique<schema::Conv2DT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->strideH = 1;
  attr->strideW = 1;
  attr->dilateH = 1;
  attr->dilateW = 1;
  attr->group = 1;
  attr->padMode = schema::PadMode_NOTSET;
  attr->format = schema::Format::Format_NCHW;

  // set opdef each attr params
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      attr->group = static_cast<int32_t>(onnx_node_attr.i());
    } else if (onnx_node_attr.name() == "dilations") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "dilations size " << onnx_node_attr.ints().size() << " is not 2";
        return nullptr;
      }
      attr->dilateH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->dilateW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernels") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return nullptr;
      }
      attr->kernelH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->kernelW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernel_shape") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return nullptr;
      }
      attr->kernelH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->kernelW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "auto_pad") {
      attr->padMode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "pads") {
      if (onnx_node_attr.ints().size() != 4) {
        MS_LOG(ERROR) << "pads size " << onnx_node_attr.ints().size() << " is not 4";
        return nullptr;
      }
      attr->padUp = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->padLeft = static_cast<int32_t>(onnx_node_attr.ints(1));
      attr->padDown = static_cast<int32_t>(onnx_node_attr.ints(2));
      attr->padRight = static_cast<int32_t>(onnx_node_attr.ints(3));
    } else if (onnx_node_attr.name() == "strides") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "strides size " << onnx_node_attr.ints().size() << " is not 2";
        return nullptr;
      }
      attr->strideH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->strideW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "order") {
      if (onnx_node_attr.s() == "NHWC") {
        attr->format = schema::Format::Format_NHWC;
      } else {
        MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s();
        return nullptr;
      }
    }
  }

  const auto &onnx_conv_weight = onnx_node.input(1);
  if (onnx_node.op_type() == "Conv") {
    auto node_iter =
      std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                   [onnx_conv_weight](const onnx::TensorProto &proto) { return proto.name() == onnx_conv_weight; });
    if (node_iter == onnx_graph.initializer().end()) {
      MS_LOG(WARNING) << "not find node: " << onnx_conv_weight;
    } else {
      std::vector<int> weight_shape;
      auto size = (*node_iter).dims_size();
      weight_shape.reserve(size);
      for (int i = 0; i < size; ++i) {
        weight_shape.emplace_back((*node_iter).dims(i));
      }
      attr->channelOut = weight_shape[0];
      attr->channelIn = weight_shape[1] * attr->group;
    }
  } else {
    auto node_iter =
      std::find_if(onnx_graph.node().begin(), onnx_graph.node().end(),
                   [onnx_conv_weight](const onnx::NodeProto &proto) { return proto.output(0) == onnx_conv_weight; });
    if (node_iter == onnx_graph.node().end()) {
      MS_LOG(ERROR) << "can not find node: " << onnx_conv_weight;
      return nullptr;
    }
    std::vector<int> dims;
    auto iter = std::find_if((*node_iter).attribute().begin(), (*node_iter).attribute().end(),
                             [](const onnx::AttributeProto &attr) { return attr.name() == "shape"; });
    if (iter != (*node_iter).attribute().end()) {
      if (iter->ints().begin() == nullptr || iter->ints().end() == nullptr) {
        MS_LOG(ERROR) << "dims insert failed";
        return nullptr;
      }
      dims.insert(dims.begin(), iter->ints().begin(), iter->ints().end());
    }
    attr->channelOut = dims.at(0);
    attr->channelIn = dims.at(3) * attr->group;
  }
  if (onnx_node.op_type() == "ConvRelu" || onnx_node.op_type() == "Int8ConvRelu") {
    attr->activationType = schema::ActivationType_RELU;
  } else {
    attr->activationType = schema::ActivationType_NO_ACTIVATION;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  if (attr->group == attr->channelIn && attr->channelIn == attr->channelOut) {
    if (!ParseGroupConvolution(attr, primitive.get())) {
      MS_LOG(ERROR) << "Convert Convolution to Depthwise failed";
      return nullptr;
    }
  } else {
    primitive->value.type = schema::PrimitiveType_Conv2D;
    primitive->value.value = attr.release();
  }
  return PrimitiveC::Create(primitive.release());
}

OnnxNodeRegistrar g_onnxConvParser("Conv", new OnnxConvParser());
OnnxNodeRegistrar g_onnxInt8ConvParser("Int8Conv", new OnnxConvParser());
OnnxNodeRegistrar g_onnxConvReluParser("ConvRelu", new OnnxConvParser());
OnnxNodeRegistrar g_onnxInt8ConvReluParser("Int8ConvRelu", new OnnxConvParser());
}  // namespace mindspore::lite
