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
#include "mindspore/lite/tools/converter/parser/onnx/onnx_relu_parser.h"
#include "securec/include/securec.h"
namespace mindspore {
namespace lite {
STATUS OnnxReluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  unique_ptr<schema::ActivationT> attr(new schema::ActivationT());
  const auto &relu_type = onnx_node.op_type();
  if (relu_type == "Relu") {
    attr->type = schema::ActivationType_RELU;
  } else if (relu_type == "LeakyRelu") {
    attr->type = schema::ActivationType_LEAKY_RELU;
  }

  if (op != nullptr) {
    op->primitive->value.type = schema::PrimitiveType_Activation;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxPReluParser::Parse(const onnx::GraphProto &onnx_graph,
                              const onnx::NodeProto &onnx_node,
                              schema::CNodeT *op) {
  if (onnx_node.input_size() != 2) {
    // MS_LOGE("input num is not 2")
    return RET_PARAM_INVALID;
  }
  unique_ptr<schema::PreluT> attr(new schema::PreluT());
  std::vector<onnx::TensorProto> params;
  for (int i = 0; i < onnx_node.input_size(); ++i) {
    const auto &input_name = onnx_node.input(i);
    for ( const auto &it : onnx_graph.initializer() ) {
      if (it.name() == "input_name") {
        params.push_back(it);
        break;
      }
    }
  }
  const onnx::TensorProto *slope = &params[0];
  if (slope == nullptr) {
    // MS_LOGE("input error")
    return RET_PARAM_INVALID;
  }
  const auto slope_raw_data = reinterpret_cast<const float *>(slope->raw_data().data());
  const int64_t slope_size = slope->raw_data().size() / sizeof(float);
  if (memcpy_s(attr->slope.data(), slope_size * sizeof(float), slope_raw_data, slope_size * sizeof(float)) != 0) {
    // MS_LOGE("memcpy_s failed")
    return RET_ERROR;
  }
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Prelu;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxReluParser("Relu", new OnnxReluParser());
OnnxNodeRegistrar g_onnxLeakyReluParser("LeakyRelu", new OnnxLeakeyReluParser());
OnnxNodeRegistrar g_onnxPReluParser("Prelu", new OnnxPReluParser());
}  // namespace lite
}  // namespace mindspore

