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

#include <vector>
#include <memory>
#include <algorithm>
#include "tools/converter/parser/onnx/onnx_swin_attention_ffn_parser.h"
#include "tools/converter/ops/ops_def.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxSwinAttentionFFNParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<SwinAttentionFFN>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::vector<int64_t> shifts;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "shifts") {
      shifts.resize(onnx_node_attr.ints_size());
      std::copy(onnx_node_attr.ints().begin(), onnx_node_attr.ints().end(), shifts.begin());
      prim->AddAttr("shifts", MakeValue(shifts));
    }
  }
  return prim;
}

OnnxNodeRegistrar g_onnxSwinAttentionFFNParser("SwinAttentionFFN", new OnnxSwinAttentionFFNParser());
}  // namespace lite
}  // namespace mindspore
