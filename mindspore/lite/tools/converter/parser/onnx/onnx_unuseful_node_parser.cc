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

#include "tools/converter/parser/onnx/onnx_unuseful_node_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS OnnxUnusefulNodeParser::Parse(const onnx::GraphProto &onnx_graph,
                                     const onnx::NodeProto &onnx_node,
                                     schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx UnusefulNodeParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  if (onnx_node.op_type() == "Int8Quantize") {
    std::unique_ptr<schema::OnnxInt8QuantizeT> attr = std::make_unique<schema::OnnxInt8QuantizeT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return RET_NULL_PTR;
    }
    op->primitive->value.type = schema::PrimitiveType_OnnxInt8Quantize;
    op->primitive->value.value = attr.release();
  } else if (onnx_node.op_type() == "Int8Dequantize") {
    std::unique_ptr<schema::OnnxInt8DequantizeT> attr = std::make_unique<schema::OnnxInt8DequantizeT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return RET_NULL_PTR;
    }
    op->primitive->value.type = schema::PrimitiveType_OnnxInt8Dequantize;
    op->primitive->value.value = attr.release();
  } else {
    MS_LOG(ERROR) << "Unsupported nodeType: " << onnx_node.op_type().c_str();
    return RET_ERROR;
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxInt8QuantizeParser("Int8Quantize", new OnnxUnusefulNodeParser());
OnnxNodeRegistrar g_onnxInt8DequantizeParser("Int8Dequantize", new OnnxUnusefulNodeParser());
}  // namespace lite
}  // namespace mindspore

