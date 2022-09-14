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
#include <string>
#include "ops/fusion/conv2d_fusion.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::lite {
STATUS GetConvChannel(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, int64_t group,
                      int64_t *channel_out, int64_t *channel_in) {
  MS_ASSERT(channel_out != nullptr);
  MS_ASSERT(channel_in != nullptr);
  MS_CHECK_GE(onnx_node.input_size(), kInputSize1, RET_ERROR);
  const auto &onnx_conv_weight = onnx_node.input(kWeightIndex);
  if (onnx_node.op_type() == "Conv") {
    auto node_iter =
      std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                   [onnx_conv_weight](const onnx::TensorProto &proto) { return proto.name() == onnx_conv_weight; });
    if (node_iter == onnx_graph.initializer().end()) {
      MS_LOG(WARNING) << "not find node: " << onnx_conv_weight;
      return RET_NO_CHANGE;
    } else {
      std::vector<int> weight_shape;
      auto size = (*node_iter).dims_size();
      weight_shape.reserve(size);
      for (int i = 0; i < size; ++i) {
        weight_shape.emplace_back((*node_iter).dims(i));
      }
      // filter of conv should have at lease two dims
      if (size < DIMENSION_2D) {
        MS_LOG(ERROR) << "index out of dims range";
        return RET_ERROR;
      }
      *channel_out = weight_shape.at(0);
      if (INT_MUL_OVERFLOW_THRESHOLD(weight_shape.at(1), group, INT64_MAX)) {
        MS_LOG(ERROR) << "channel in overflow";
        return RET_ERROR;
      }
      *channel_in = weight_shape.at(1) * group;
    }
  } else {
    auto node_iter =
      std::find_if(onnx_graph.node().begin(), onnx_graph.node().end(),
                   [onnx_conv_weight](const onnx::NodeProto &proto) { return proto.output(0) == onnx_conv_weight; });
    if (node_iter == onnx_graph.node().end()) {
      MS_LOG(ERROR) << "can not find node: " << onnx_conv_weight;
      return RET_ERROR;
    }
    std::vector<int> dims;
    auto iter = std::find_if((*node_iter).attribute().begin(), (*node_iter).attribute().end(),
                             [](const onnx::AttributeProto &attr) { return attr.name() == "shape"; });
    if (iter != (*node_iter).attribute().end()) {
      if (iter->ints().begin() == nullptr || iter->ints().end() == nullptr) {
        MS_LOG(ERROR) << "dims insert failed";
        return RET_ERROR;
      }
      dims.insert(dims.begin(), iter->ints().begin(), iter->ints().end());
    } else {
      return RET_NO_CHANGE;
    }
    // filter of conv should have at lease four dims
    if (dims.size() < DIMENSION_4D) {
      MS_LOG(ERROR) << "index out of dims range";
      return RET_ERROR;
    }
    *channel_out = dims.at(kNHWC_N);
    // the fourth dim of filter of conv is channel dim
    if (INT_MUL_OVERFLOW_THRESHOLD(dims.at(kNHWC_C), group, INT64_MAX)) {
      MS_LOG(ERROR) << "channel in overflow";
      return RET_ERROR;
    }
    *channel_in = dims.at(kNHWC_C) * group;
  }
  return RET_OK;
}

STATUS OnnxConvParser::ParseOnnxAttr(const onnx::NodeProto &onnx_node, int64_t *group, mindspore::Format *format,
                                     mindspore::PadMode *pad_mode) {
  MS_ASSERT(group != nullptr);
  MS_ASSERT(format != nullptr);
  MS_ASSERT(pad_mode != nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      *group = onnx_node_attr.i();
    } else if (onnx_node_attr.name() == "auto_pad") {
      *pad_mode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "order" && onnx_node_attr.s() != "NHWC") {
      MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s();
      return RET_ERROR;
    } else if (onnx_node_attr.name() == "order") {
      if (onnx_node_attr.s() == "NHWC") {
        *format = mindspore::Format::NHWC;
      } else {
        MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s();
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

PrimitiveCPtr OnnxConvParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Conv2DFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  prim->set_pad({0, 0, 0, 0});
  mindspore::Format format = mindspore::Format::NCHW;
  mindspore::PadMode pad_mode = mindspore::PadMode::PAD;
  int64_t channel_out = 1;
  int64_t channel_in = 1;
  int64_t group = 1;
  std::vector<int64_t> kernels, strides, dilation, pads;
  auto status = ParseOnnxAttr(onnx_node, &group, &format, &pad_mode);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Parse onnx attribute failed.";
    return nullptr;
  }
  (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(format));
  prim->set_pad_mode(pad_mode);
  prim->set_group(group);

  bool conv1d = false;
  if (ParseVecAttr(onnx_node, &kernels, &strides, &dilation, &pads, &conv1d) != RET_OK) {
    return nullptr;
  }
  if (conv1d) {
    (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(NCW));
  }
  prim->set_dilation({1, 1});
  if (!dilation.empty()) {
    prim->set_dilation(dilation);
  }
  prim->set_pad_list({0, 0, 0, 0});
  if (!pads.empty()) {
    prim->set_pad_list(pads);
  }
  if (!kernels.empty()) {
    prim->set_kernel_size(kernels);
  }
  if (!strides.empty()) {
    prim->set_stride(strides);
  }

  // get channel_out and channel_in
  status = GetConvChannel(onnx_graph, onnx_node, group, &channel_out, &channel_in);
  if (status == RET_OK) {
    prim->set_in_channel(channel_in);
    prim->set_out_channel(channel_out);
  } else if (status != RET_NO_CHANGE) {
    return nullptr;
  }

  // parse activationType
  prim->set_activation_type(mindspore::ActivationType::NO_ACTIVATION);
  if (onnx_node.op_type() == "ConvRelu" || onnx_node.op_type() == "Int8ConvRelu") {
    prim->set_activation_type(mindspore::ActivationType::RELU);
  }

  if (group == channel_in && channel_in == channel_out) {
    (void)prim_c->AddAttr(ops::kIsDepthWise, MakeValue<bool>(true));
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxConvParser("Conv", new OnnxConvParser());
OnnxNodeRegistrar g_onnxInt8ConvParser("Int8Conv", new OnnxConvParser());
OnnxNodeRegistrar g_onnxConvReluParser("ConvRelu", new OnnxConvParser());
OnnxNodeRegistrar g_onnxInt8ConvReluParser("Int8ConvRelu", new OnnxConvParser());
}  // namespace mindspore::lite
