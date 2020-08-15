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

#include <memory>
#include "tools/converter/parser/onnx/onnx_lrn_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxLrnParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx LrnParser";
  std::unique_ptr<schema::LrnT> attr(new schema::LrnT());
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto& attribute_name = onnx_node_attr.name();
    if (attribute_name == "size") {
      attr->size = static_cast<int32_t>(onnx_node_attr.i());
    } else if (attribute_name == "alpha") {
      attr->alpha = onnx_node_attr.f();
    } else if (attribute_name == "beta") {
      attr->beta = onnx_node_attr.f();
    } else if (attribute_name == "bias") {
      attr->bias = onnx_node_attr.f();
    }
  }
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Lrn;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxLrnxParser("Lrn", new OnnxLrnParser());
}  // namespace lite
}  // namespace mindspore

