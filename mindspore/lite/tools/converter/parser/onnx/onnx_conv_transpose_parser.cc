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

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxDeConvParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Conv2dTransposeFusion>();

  prim->set_pad({0, 0, 0, 0});
  mindspore::Format format = mindspore::Format::NCHW;
  mindspore::PadMode pad_mode = mindspore::PadMode::PAD;

  int64_t group = 1;
  std::vector<int64_t> kernel;
  std::vector<int64_t> dilate;
  std::vector<int64_t> stride;
  std::vector<int64_t> pads;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      group = onnx_node_attr.i();
    } else if (onnx_node_attr.name() == "dilations") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "dilations size " << onnx_node_attr.ints().size() << " is not 2";
        return nullptr;
      }
      dilate.push_back(onnx_node_attr.ints(0));
      dilate.push_back(onnx_node_attr.ints(1));
      prim->set_dilation(dilate);
    } else if (onnx_node_attr.name() == "kernels") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return nullptr;
      }
      kernel.push_back(onnx_node_attr.ints(0));
      kernel.push_back(onnx_node_attr.ints(1));
      prim->set_kernel_size(kernel);
    } else if (onnx_node_attr.name() == "kernel_shape") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return nullptr;
      }
      kernel.push_back(onnx_node_attr.ints(0));
      kernel.push_back(onnx_node_attr.ints(1));
      prim->set_kernel_size(kernel);
    } else if (onnx_node_attr.name() == "auto_pad") {
      pad_mode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "pads") {
      if (onnx_node_attr.ints().size() != 4) {
        MS_LOG(ERROR) << "pads size " << onnx_node_attr.ints().size() << " is not 4";
        return nullptr;
      }
      pads.push_back(onnx_node_attr.ints(0));
      pads.push_back(onnx_node_attr.ints(2));
      pads.push_back(onnx_node_attr.ints(1));
      pads.push_back(onnx_node_attr.ints(3));
      prim->set_pad_list(pads);
    } else if (onnx_node_attr.name() == "strides") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "strides size " << onnx_node_attr.ints().size() << " is not 2";
        return nullptr;
      }
      stride.push_back(onnx_node_attr.ints(0));
      stride.push_back(onnx_node_attr.ints(1));
      prim->set_stride(stride);
    } else if (onnx_node_attr.name() == "order") {
      if (onnx_node_attr.s() == "NHWC") {
        format = mindspore::Format::NHWC;
      } else {
        MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s().c_str();
        return nullptr;
      }
    } else if (onnx_node_attr.name() == "output_padding") {
      MS_LOG(ERROR) << "output_padding param hasn't been supported";
      return nullptr;
    }
  }
  prim->set_format(format);
  prim->set_group(group);
  prim->set_pad_mode(pad_mode);

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
