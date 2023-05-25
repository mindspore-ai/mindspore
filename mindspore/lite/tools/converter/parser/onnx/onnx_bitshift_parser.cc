/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_bitshift_parser.h"
#include <memory>
#include <string>
#include "ops/left_shift.h"
#include "ops/rightshift.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxBitShiftParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  std::string direction_s = "";
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "direction") {
      const auto &direction = onnx_node_attr.s();
      direction_s = direction;
    }
  }
  if (direction_s == "LEFT") {
    auto prim = std::make_unique<ops::LeftShift>();
    MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
    return prim->GetPrim();
  } else {
    auto prim = std::make_unique<ops::RightShift>();
    MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
    return prim->GetPrim();
  }
}

OnnxNodeRegistrar g_onnxBitShiftParser("BitShift", new OnnxBitShiftParser());
}  // namespace lite
}  // namespace mindspore
