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

#include "tools/converter/parser/onnx/onnx_shrink_parser.h"
#include <memory>
#include "tools/converter/ops/ops_def.h"
#include "ir/value.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxShrinkParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<lite::Shrink>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Shrink prim failed.";
    return nullptr;
  }
  float bias = 0.0;
  float lambd = 0.5;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "bias") {
      bias = onnx_node_attr.f();
    } else if (onnx_node_attr.name() == "lambd") {
      lambd = onnx_node_attr.f();
    }
  }
  prim->set_attr("bias", MakeValue(bias));
  prim->set_attr("lambd", MakeValue(lambd));

  return prim;
}
OnnxNodeRegistrar g_onnxShrinkParser("Shrink", new OnnxShrinkParser());
}  // namespace lite
}  // namespace mindspore
