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
  auto primitive_c = new (std::nothrow) ops::Activation;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new ReLU failed";
    return nullptr;
  }

  primitive_c->set_activation_type(mindspore::ActivationType::RELU);

  return primitive_c;
}

ops::PrimitiveC *OnnxLeakyReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Activation;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new LeakyRelu failed";
    return nullptr;
  }

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "alpha") {
      primitive_c->set_alpha(onnx_node_attr.f());
    }
  }

  primitive_c->set_activation_type(mindspore::ActivationType::LEAKY_RELU);

  return primitive_c;
}

ops::PrimitiveC *OnnxPReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::PReLUFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new PReLU failed";
    return nullptr;
  }

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
    const auto slope_raw_data = reinterpret_cast<const float *>(slope_data->raw_data().data());
    const int64_t slope_size = slope_data->raw_data().size() / sizeof(float);
    std::vector<float> slope;
    bool channelShared = false;
    if (slope_size == 1) {
      slope.push_back(*slope_raw_data);
      channelShared = true;
    } else {
      slope.resize(slope_size);
      if (memcpy_s(slope.data(), slope_size * sizeof(float), slope_raw_data, slope_size * sizeof(float)) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed";
        return nullptr;
      }
    }
    primitive_c->set_slope(slope);
    primitive_c->set_channel_shared(channelShared);
  } else {
    MS_LOG(WARNING) << "The slope pf prelu is null, which may cause errors.";
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxEluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Elu;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Elu failed";
    return nullptr;
  }

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "alpha") {
      primitive_c->set_alpha(onnx_node_attr.f());
    }
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxTanhParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Activation;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Tanh failed";
    return nullptr;
  }

  primitive_c->set_activation_type(mindspore::ActivationType::TANH);

  return primitive_c;
}

ops::PrimitiveC *OnnxSigmoidParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Activation;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Sigmoid failed";
    return nullptr;
  }

  primitive_c->set_activation_type(mindspore::ActivationType::SIGMOID);

  return primitive_c;
}

OnnxNodeRegistrar g_onnxReluParser("Relu", new OnnxReluParser());
OnnxNodeRegistrar g_onnxLeakyReluParser("LeakyRelu", new OnnxLeakyReluParser());
OnnxNodeRegistrar g_onnxPReluParser("PRelu", new OnnxPReluParser());
OnnxNodeRegistrar g_onnxEluParser("Elu", new OnnxEluParser());
OnnxNodeRegistrar g_onnxTanhParser("Tanh", new OnnxTanhParser());
OnnxNodeRegistrar g_onnxSigmoodParser("Sigmoid", new OnnxSigmoidParser());
}  // namespace lite
}  // namespace mindspore
