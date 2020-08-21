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
#include <vector>
#include <memory>
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
STATUS OnnxReluParser::Parse(const onnx::GraphProto &onnx_graph,
                             const onnx::NodeProto &onnx_node,
                             schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx ReluParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ActivationT> attr = std::make_unique<schema::ActivationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const auto &relu_type = onnx_node.op_type();
  if (relu_type == "Relu") {
    MS_LOG(DEBUG) << "onnx ReluParser";
    attr->type = schema::ActivationType_RELU;
  } else if (relu_type == "LeakyRelu") {
    MS_LOG(DEBUG) << "onnx LeakyReluParser";
    attr->type = schema::ActivationType_LEAKY_RELU;
  }

  op->primitive->value.type = schema::PrimitiveType_Activation;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS OnnxPReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                              schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx PReluParser";

  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  if (onnx_node.input_size() != 2) {
    MS_LOG(ERROR) << "input num should be 2";
    return RET_ERROR;
  }
  std::unique_ptr<schema::CaffePReLUT> attr = std::make_unique<schema::CaffePReLUT>();
  std::vector<onnx::TensorProto> params;
  const auto &input_name = onnx_node.input(1);
  for (const auto &it : onnx_graph.initializer()) {
    if (it.name() == input_name) {
      params.push_back(it);
      break;
    }
  }

  const onnx::TensorProto *slope = &params[0];
  if (slope == nullptr) {
    MS_LOG(ERROR) << "input error: params[0] is null";
    return RET_ERROR;
  }
  const auto slope_raw_data = reinterpret_cast<const float *>(slope->raw_data().data());
  const int64_t slope_size = slope->raw_data().size() / sizeof(float);
  if (slope_size == 1) {
    attr->slope.push_back(*slope_raw_data);
    attr->channelShared = true;
  } else {
    attr->slope.resize(slope_size);
    attr->channelShared = false;
    if (memcpy_s(attr->slope.data(), slope_size * sizeof(float), slope_raw_data, slope_size * sizeof(float)) != 0) {
      MS_LOG(ERROR) << "memcpy_s failed";
      return RET_ERROR;
    }
  }

  op->primitive->value.type = schema::PrimitiveType_CaffePReLU;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxReluParser("Relu", new OnnxReluParser());
OnnxNodeRegistrar g_onnxLeakyReluParser("LeakyRelu", new OnnxLeakeyReluParser());
OnnxNodeRegistrar g_onnxPReluParser("PRelu", new OnnxPReluParser());
}  // namespace lite
}  // namespace mindspore
