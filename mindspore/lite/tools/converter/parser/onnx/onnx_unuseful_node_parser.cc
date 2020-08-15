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
#include "tools/converter/parser/onnx/onnx_unuseful_node_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxUnusefulNodeParser::Parse(const onnx::GraphProto &onnx_graph,
                                     const onnx::NodeProto &onnx_node,
                                     schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx UnusefulNodeParser";
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    if (onnx_node.op_type() == "Int8Quantize") {
      op->primitive->value.type = schema::PrimitiveType_OnnxInt8Quantize;
      op->primitive->value.value = new (std::nothrow) schema::OnnxInt8QuantizeT;
    } else if (onnx_node.op_type() == "Int8Dequantize") {
      op->primitive->value.type = schema::PrimitiveType_OnnxInt8Dequantize;
      op->primitive->value.value = new (std::nothrow) schema::OnnxInt8DequantizeT;
    } else {
      // MS_LOGE("Unsupported nodeType: %s", onnx_node.op_type().c_str());
      return RET_ERROR;
    }
    if (op->primitive->value.value == nullptr) {
      // MS_LOGE("new %s attr value failed", onnx_node.op_type().c_str());
      return RET_ERROR;
    }
  } else {
    // MS_LOGE("Input opDef is nullptr");
    return RET_PARAM_INVALID;
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxInt8QuantizeParser("Int8Quantize", new OnnxUnusefulNodeParser());
OnnxNodeRegistrar g_onnxInt8DequantizeParser("Int8Dequantize", new OnnxUnusefulNodeParser());
}  // namespace lite
}  // namespace mindspore

