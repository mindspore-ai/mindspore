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

#include "tools/converter/parser/onnx/onnx_activation_parser.h"
#include <memory>
#include <vector>
#include "securec/include/securec.h"
#include "ops/fusion/prelu_fusion.h"
#include "ops/elu.h"
#include "ops/fusion/activation.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::RELU);

  return prim.release();
}

ops::PrimitiveC *OnnxLeakyReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "alpha") {
      prim->set_alpha(onnx_node_attr.f());
    }
  }

  prim->set_activation_type(mindspore::ActivationType::LEAKY_RELU);

  return prim.release();
}

ops::PrimitiveC *OnnxPReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::PReLUFusion>();

  std::vector<onnx::TensorProto> params;
  const auto &input_name = onnx_node.input(1);
  auto node_iter = std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                                [input_name](const onnx::TensorProto &proto) { return proto.name() == input_name; });
  if (node_iter == onnx_graph.initializer().end()) {
    MS_LOG(ERROR) << "not find node: " << input_name.c_str();
    return nullptr;
  } else {
    params.push_back(*node_iter);
  }

  if (!params.empty()) {
    const onnx::TensorProto *slope_data = &params[0];
    if (slope_data == nullptr) {
      MS_LOG(ERROR) << "input error: params[0] is null";
      return nullptr;
    }
    std::vector<float> slope;
    if (slope_data->float_data_size() > 0) {
      const int64_t slope_size = slope_data->float_data_size();
      for (int64_t i = 0; i < slope_size; i++) {
        slope.emplace_back(slope_data->float_data(i));
      }
      prim->set_slope(slope);
      prim->set_channel_shared(slope_size == 1);
    } else {
      const auto slope_raw_data = reinterpret_cast<const float *>(slope_data->raw_data().data());
      const int64_t slope_size = slope_data->raw_data().size() / sizeof(float);
      slope.resize(slope_size);
      bool channel_shared = false;
      if (slope_size == 1) {
        slope.push_back(*slope_raw_data);
        channel_shared = true;
      } else {
        slope.resize(slope_size);
        if (memcpy_s(slope.data(), slope_size * sizeof(float), slope_raw_data, slope_size * sizeof(float)) != EOK) {
          MS_LOG(ERROR) << "memcpy_s failed";
          return nullptr;
        }
      }
      prim->set_slope(slope);
      prim->set_channel_shared(channel_shared);
    }
  } else {
    MS_LOG(WARNING) << "The slope pf prelu is null, which may cause errors.";
  }

  return prim.release();
}

ops::PrimitiveC *OnnxEluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Elu>();

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "alpha") {
      prim->set_alpha(onnx_node_attr.f());
    }
  }

  return prim.release();
}

ops::PrimitiveC *OnnxTanhParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::TANH);

  return prim.release();
}

ops::PrimitiveC *OnnxSigmoidParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Activation>();

  prim->set_activation_type(mindspore::ActivationType::SIGMOID);

  return prim.release();
}

OnnxNodeRegistrar g_onnxReluParser("Relu", new OnnxReluParser());
OnnxNodeRegistrar g_onnxLeakyReluParser("LeakyRelu", new OnnxLeakyReluParser());
OnnxNodeRegistrar g_onnxPReluParser("PRelu", new OnnxPReluParser());
OnnxNodeRegistrar g_onnxEluParser("Elu", new OnnxEluParser());
OnnxNodeRegistrar g_onnxTanhParser("Tanh", new OnnxTanhParser());
OnnxNodeRegistrar g_onnxSigmoodParser("Sigmoid", new OnnxSigmoidParser());
}  // namespace lite
}  // namespace mindspore
