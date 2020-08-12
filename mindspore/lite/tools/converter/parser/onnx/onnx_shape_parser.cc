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
#include "tools/converter/parser/onnx/onnx_shape_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxShapeParser::Parse(const onnx::GraphProto &onnx_graph,
                              const onnx::NodeProto &onnx_node,
                              schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx ShapeParser";
  if (op != nullptr) {
    std::unique_ptr<schema::ShapeT> attr(new schema::ShapeT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Shape;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxShapeParser("Shape", new OnnxShapeParser());
}  // namespace lite
}  // namespace mindspore

