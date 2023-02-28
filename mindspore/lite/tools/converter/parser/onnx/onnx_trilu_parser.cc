/**
 * Copyright 2023
 * Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_trilu_parser.h"
#include <memory>
#include "ops/tril.h"
#include "ops/triu.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxTriluParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  bool upper = true;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "upper") {
      upper = onnx_node_attr.i() != 0;
    }
  }
  if (upper) {
    auto prim = std::make_unique<ops::Triu>();
    MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
    return prim->GetPrim();
  } else {
    auto prim = std::make_unique<ops::Tril>();
    MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
    return prim->GetPrim();
  }
}

OnnxNodeRegistrar g_onnxTriluParser("Trilu", new OnnxTriluParser());
}  // namespace lite
}  // namespace mindspore
