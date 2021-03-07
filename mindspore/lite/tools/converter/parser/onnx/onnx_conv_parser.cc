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

namespace mindspore::lite {
STATUS ParseVecAttr(const onnx::NodeProto &onnx_node, std::vector<int64_t> *kernels, std::vector<int64_t> *strides,
                    std::vector<int64_t> *dilation, std::vector<int64_t> *pads) {
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "dilations") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "dilations size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      dilation->push_back(onnx_node_attr.ints(0));
      dilation->push_back(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernels") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      kernels->push_back(onnx_node_attr.ints(0));
      kernels->push_back(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernel_shape") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      kernels->push_back(onnx_node_attr.ints(0));
      kernels->push_back(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "pads") {
      if (onnx_node_attr.ints().size() != 4) {
        MS_LOG(ERROR) << "pads size " << onnx_node_attr.ints().size() << " is not 4";
        return RET_ERROR;
      }
      pads->push_back(onnx_node_attr.ints(0));
      pads->push_back(onnx_node_attr.ints(2));
      pads->push_back(onnx_node_attr.ints(1));
      pads->push_back(onnx_node_attr.ints(3));
    } else if (onnx_node_attr.name() == "strides") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "strides size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      strides->push_back(onnx_node_attr.ints(0));
      strides->push_back(onnx_node_attr.ints(1));
    }
  }
  return RET_OK;
}

STATUS GetConvChannel(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, int64_t group,
                      int64_t *channel_out, int64_t *channel_in) {
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
      *channel_out = weight_shape[0];
      *channel_in = weight_shape[1] * group;
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
    }
    *channel_out = dims.at(0);
    *channel_in = dims.at(3) * group;
  }
  return RET_OK;
}

ops::PrimitiveC *OnnxConvParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Conv2DFusion>();

  prim->set_pad({0, 0, 0, 0});
  mindspore::Format format = mindspore::Format::NCHW;
  mindspore::PadMode pad_mode = mindspore::PadMode::PAD;
  int64_t channel_out = 1, channel_in = 1, group = 1;
  std::vector<int64_t> kernels, strides, dilation, pads;

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      group = onnx_node_attr.i();
    } else if (onnx_node_attr.name() == "auto_pad") {
      pad_mode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "order" && onnx_node_attr.s() != "NHWC") {
      MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s();
      return nullptr;
    } else if (onnx_node_attr.name() == "order") {
      if (onnx_node_attr.s() == "NHWC") {
        format = mindspore::Format::NHWC;
      } else {
        MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s();
        return nullptr;
      }
    }
  }
  prim->set_format(format);
  prim->set_pad_mode(pad_mode);
  prim->set_group(group);

  if (ParseVecAttr(onnx_node, &kernels, &strides, &dilation, &pads) != RET_OK) {
    return nullptr;
  }
  if (dilation.empty()) {
    prim->set_dilation({1, 1});
  } else {
    prim->set_dilation(dilation);
  }
  if (pads.empty()) {
    prim->set_pad_list({0, 0, 0, 0});
  } else {
    prim->set_pad_list(pads);
  }
  if (!kernels.empty()) {
    prim->set_kernel_size(kernels);
  }
  if (!strides.empty()) {
    prim->set_stride(strides);
  }

  // get channel_out and channel_in
  if (GetConvChannel(onnx_graph, onnx_node, group, &channel_out, &channel_in) != RET_OK) {
    return nullptr;
  }
  prim->set_in_channel(channel_in);
  prim->set_out_channel(channel_out);

  // parse activationType
  if (onnx_node.op_type() == "ConvRelu" || onnx_node.op_type() == "Int8ConvRelu") {
    prim->set_activation_type(mindspore::ActivationType::RELU);
  } else {
    prim->set_activation_type(mindspore::ActivationType::NO_ACTIVATION);
  }

  if (group == channel_in && channel_in == channel_out) {
    prim->AddAttr(ops::kIsDepthWise, MakeValue<bool>(true));
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxConvParser("Conv", new OnnxConvParser());
OnnxNodeRegistrar g_onnxInt8ConvParser("Int8Conv", new OnnxConvParser());
OnnxNodeRegistrar g_onnxConvReluParser("ConvRelu", new OnnxConvParser());
OnnxNodeRegistrar g_onnxInt8ConvReluParser("Int8ConvRelu", new OnnxConvParser());
}  // namespace mindspore::lite
