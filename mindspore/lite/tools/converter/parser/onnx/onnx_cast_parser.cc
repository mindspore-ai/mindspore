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

#include "tools/converter/parser/onnx/onnx_cast_parser.h"
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include <memory>
#include "ops/cast.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxCastParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Cast>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "to") {
      auto dst_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(onnx_node_attr.i()));
      if (dst_type == kNumberTypeInt64) {
        dst_type = kNumberTypeInt32;
      }
      if (dst_type == kNumberTypeFloat64) {
        dst_type = kNumberTypeFloat32;
      }
      auto prim_c = prim->GetPrim();
      MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
      (void)prim_c->AddAttr("to", MakeValue(static_cast<int32_t>(dst_type)));
    }
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxCastParser("Cast", new OnnxCastParser());
}  // namespace lite
}  // namespace mindspore
