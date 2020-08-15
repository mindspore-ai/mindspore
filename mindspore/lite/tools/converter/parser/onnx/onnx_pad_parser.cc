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
#include "tools/converter/parser/onnx/onnx_pad_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxPadParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx PadParser";
  std::unique_ptr<schema::PadT> attr(new schema::PadT());
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "pads") {
      const int size = onnx_node_attr.ints_size();
      attr->paddings.resize(size);
      for (int i = 0; i < size / 2; ++i) {
        attr->paddings[i * 2] = static_cast<int32_t>(onnx_node_attr.ints(i));
        attr->paddings[i * 2 + 1] = static_cast<int32_t>(onnx_node_attr.ints(i + size / 2));
      }
    } else if (attribute_name == "mode") {
      const auto &mode =  onnx_node_attr.s();
      if  (mode == "constant") {
        attr->paddingMode = schema::PaddingMode_CONSTANT;
      } else if (mode == "reflect") {
        attr->paddingMode = schema::PaddingMode_REFLECT;
      } else if (mode == "edge") {
        attr->paddingMode = schema::PaddingMode_SYMMETRIC;
      }
    }
  }
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Pad;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxPadParser("Pad", new OnnxPadParser());
}  // namespace lite
}  // namespace mindspore

