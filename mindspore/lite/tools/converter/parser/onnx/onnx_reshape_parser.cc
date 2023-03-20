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

#include "tools/converter/parser/onnx/onnx_reshape_parser.h"
#include <vector>
#include <memory>
#include "ops/reshape.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxReshapeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Reshape>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  std::vector<int32_t> shape;
  shape.clear();
  if (onnx_node.input_size() != 2) {
    for (const auto &onnx_node_attr : onnx_node.attribute()) {
      const auto &attribute_name = onnx_node_attr.name();
      if (attribute_name == "shape") {
        for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
          shape.push_back(static_cast<int>(onnx_node_attr.ints(i)));
        }
        (void)prim_c->AddAttr("shape", MakeValue(shape));
      }
    }
  }
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "allowzero") {
      int allowzero = onnx_node_attr.i();
      (void)prim_c->AddAttr("allowzero", MakeValue(allowzero));
    }
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxReshapeParser("Reshape", new OnnxReshapeParser());
}  // namespace lite
}  // namespace mindspore
