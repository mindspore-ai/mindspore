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

#include "tools/converter/parser/onnx/onnx_conv_transpose_parser.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "tools/converter/parser/onnx/onnx_conv_parser.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxDeConvParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Conv2dTransposeFusion>();

  prim->set_pad({0, 0, 0, 0});
  mindspore::PadMode pad_mode = mindspore::PadMode::PAD;
  std::vector<int64_t> kernel, dilate, stride, pads, output_paddings;
  int64_t group = 1;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      group = onnx_node_attr.i();
    } else if (onnx_node_attr.name() == "auto_pad") {
      pad_mode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "order" && onnx_node_attr.s() != "NHWC") {
      MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s().c_str();
      return nullptr;
    }
    if (onnx_node_attr.name() == "output_padding") {
      output_paddings.push_back(static_cast<int32_t>(onnx_node_attr.ints(0)));
      output_paddings.push_back(static_cast<int32_t>(onnx_node_attr.ints(1)));
      prim->set_output_paddings(output_paddings);
    }
  }
  prim->set_format(mindspore::Format::NCHW);
  prim->set_group(group);
  prim->set_pad_mode(pad_mode);

  if (ParseVecAttr(onnx_node, &kernel, &stride, &dilate, &pads) != RET_OK) {
    return nullptr;
  }
  if (!dilate.empty()) {
    prim->set_dilation(dilate);
  }
  if (!pads.empty()) {
    prim->set_pad_list(pads);
  }
  if (!kernel.empty()) {
    prim->set_kernel_size(kernel);
  }
  if (!stride.empty()) {
    prim->set_stride(stride);
  }
  if (output_paddings.empty()) {
    prim->set_output_paddings({0, 0});
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
  prim->set_in_channel(weight_shape[0]);
  prim->set_out_channel(weight_shape[1] * group);

  if (group != 1 && weight_shape[1] == 1) {
    prim->AddAttr(ops::kIsDepthWise, MakeValue<bool>(true));
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxDeConvParser("ConvTranspose", new OnnxDeConvParser());
}  // namespace lite
}  // namespace mindspore
