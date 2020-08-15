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

#include <vector>
#include <memory>
#include <algorithm>
#include "tools/converter/parser/onnx/onnx_deconv_parser.h"

namespace mindspore {
namespace lite {
bool OnnxDeConvParser::ParseGroupDeConvolution(schema::CNodeT *op, schema::DeConv2DT *attr) {
  MS_LOG(DEBUG) << "onnx DeConvParser";
  if (attr == nullptr || attr->group != attr->channelOut) {
    return false;
  }
  auto deDepthwiseConv2DParam(new (std::nothrow) schema::DeDepthwiseConv2DT());
  if (deDepthwiseConv2DParam == nullptr) {
    // MS_LOGW("new DeDepthwiseConv2DT failed");
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
  deDepthwiseConv2DParam->hasBias = attr->hasBias;
  deDepthwiseConv2DParam->activationType = attr->activationType;
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
    delete (op->primitive->value.value);
    op->primitive->value.value = deDepthwiseConv2DParam;
  }
  return true;
}

STATUS OnnxDeConvParser::Parse(const onnx::GraphProto &onnx_graph,
                               const onnx::NodeProto &onnx_node,
                               schema::CNodeT *op) {
  auto attr = new schema::DeConv2DT();
  // set opdef each attr params
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      attr->group = static_cast<int32_t>(onnx_node_attr.i());
    } else if (onnx_node_attr.name() == "dilations") {
      if (onnx_node_attr.ints().size() != 2) {
        // MS_LOGE("dilations size %d is not 2", onnx_node_attr.ints().size());
        return RET_ERROR;
      }
      attr->dilateW = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->dilateH = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernels") {
      if (onnx_node_attr.ints().size() != 2) {
        // MS_LOGE("kernel_shape size %d is not 2", onnx_node_attr.ints().size());
        return RET_ERROR;
      }
      attr->kernelH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->kernelW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernel_shape") {
      if (onnx_node_attr.ints().size() != 2) {
        // MS_LOGE("kernel_shape size %d is not 2", onnx_node_attr.ints().size());
        return RET_ERROR;
      }
      attr->kernelW = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->kernelH = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "auto_pad") {
      attr->padMode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "pads") {
      if (onnx_node_attr.ints().size() != 4) {
        // MS_LOGE("pads size %d is not 4", onnx_node_attr.ints().size());
        return RET_ERROR;
      }
      attr->padUp = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->padLeft = static_cast<int32_t>(onnx_node_attr.ints(1));
      attr->padDown = static_cast<int32_t>(onnx_node_attr.ints(2));
      attr->padRight = static_cast<int32_t>(onnx_node_attr.ints(3));
    } else if (onnx_node_attr.name() == "strides") {
      if (onnx_node_attr.ints().size() != 2) {
        // MS_LOGE("strides size %d is not 2", onnx_node_attr.ints().size());
        return RET_ERROR;
      }
      attr->strideW = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->strideH = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "order") {
      if (onnx_node_attr.s() == "NHWC") {
        attr->format = schema::Format_NHWC;
      } else {
        // MS_LOGE("Unsupported format: %s", onnx_node_attr.s().c_str());
        return RET_ERROR;
      }
    }
  }

  const auto &onnx_conv_weight = onnx_node.input(1);
  auto nodeIter =
    std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                 [onnx_conv_weight](const onnx::TensorProto &proto) { return proto.name() == onnx_conv_weight; });
  if (nodeIter == onnx_graph.initializer().end()) {
    // MS_LOGE("not find node: %s", onnx_conv_weight.c_str())
    return RET_ERROR;
  }
  std::vector<int> weight_shape;
  auto size = (*nodeIter).dims_size();
  for (int i = 0; i < size; ++i) {
    weight_shape.emplace_back((*nodeIter).dims(i));
  }
  MS_ASSERT(weight_shape.size() == 4);
  attr->channelIn = weight_shape[0];
  attr->channelOut = weight_shape[1] * attr->group;

  attr->format = schema::Format_NCHW;
  attr->hasBias = onnx_node.input().size() == 3;

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_DeConv2D;
    op->primitive->value.value = attr;
  }

  if (attr->group != 1) {
    if (!ParseGroupDeConvolution(op, attr)) {
      delete attr;
      // MS_LOGE("Convert DeConvolution to DeDepthwise failed");
      return RET_ERROR;
    }
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxDeConvParser("ConvTranspose", new OnnxDeConvParser());
}  // namespace lite
}  // namespace mindspore

