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

#include "tools/converter/parser/onnx/onnx_relu_parser.h"
#include <memory>
#include <vector>
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
lite::PrimitiveC *OnnxReluParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx ReluParser";
  auto attr = std::make_unique<schema::ActivationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const auto &relu_type = onnx_node.op_type();
  if (relu_type == "Relu") {
    MS_LOG(DEBUG) << "onnx ReluParser";
    attr->type = schema::ActivationType_RELU;
  } else if (relu_type == "LeakyRelu") {
    MS_LOG(DEBUG) << "onnx LeakyReluParser";
    attr->type = schema::ActivationType_LEAKY_RELU;
  }
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "alpha") {
      attr->alpha = onnx_node_attr.f();
    }
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Activation;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxPReluParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                      const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx PReluParser";
  if (onnx_node.input_size() != 2) {
    MS_LOG(ERROR) << "input num should be 2";
    return nullptr;
  }
  auto attr = std::make_unique<schema::PReLUT>();
  std::vector<onnx::TensorProto> params;
  const auto &input_name = onnx_node.input(1);
  for (const auto &it : onnx_graph.initializer()) {
    if (it.name() == input_name) {
      params.push_back(it);
      break;
    }
  }

  if (!params.empty()) {
    const onnx::TensorProto *slope = &params[0];
    if (slope == nullptr) {
      MS_LOG(ERROR) << "input error: params[0] is null";
      return nullptr;
    }
    if (slope->float_data_size() > 0) {
      const int64_t slope_size = slope->float_data_size();
      for (int64_t i = 0; i < slope_size; i++) {
        attr->slope.emplace_back(slope->float_data(i));
      }
      attr->channelShared = slope_size == 1;
    } else {
      const auto slope_raw_data = reinterpret_cast<const float *>(slope->raw_data().data());
      const int64_t slope_size = slope->raw_data().size() / sizeof(float);
      attr->slope.resize(slope_size);
      if (memcpy_s(attr->slope.data(), slope_size * sizeof(float), slope_raw_data, slope_size * sizeof(float)) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed";
        return nullptr;
      }
      attr->channelShared = slope_size == 1;
    }
  } else {
    MS_LOG(WARNING) << "The slope pf prelu is null, which may cause errors.";
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_PReLU;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

OnnxNodeRegistrar g_onnxReluParser("Relu", new OnnxReluParser());
OnnxNodeRegistrar g_onnxLeakyReluParser("LeakyRelu", new OnnxReluParser());
OnnxNodeRegistrar g_onnxPReluParser("PRelu", new OnnxPReluParser());
}  // namespace lite
}  // namespace mindspore
