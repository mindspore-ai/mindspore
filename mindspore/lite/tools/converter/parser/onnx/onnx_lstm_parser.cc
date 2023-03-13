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

#include "tools/converter/parser/onnx/onnx_lstm_parser.h"
#include <memory>
#include "ops/lstm.h"
#include "nnacl/op_base.h"
#include "include/registry/converter_context.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxLstmParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LSTM>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "direction") {
      const auto &direction = onnx_node_attr.s();
      bool bidirectional = direction == "bidirectional";
      prim->set_bidirectional(bidirectional);
      if (bidirectional) {
        prim->set_num_directions(2);
      } else {
        prim->set_num_directions(1);
      }
    } else if (onnx_node_attr.name() == "hidden_size") {
      prim->set_hidden_size(onnx_node_attr.i());
    } else if (onnx_node_attr.name() == "clip") {
      prim->set_dropout(onnx_node_attr.f());
    } else if (onnx_node_attr.name() == "activations") {
      prim->set_has_bias(true);
    }
  }
  int fmk_type = mindspore::converter::FmkType::kFmkTypeOnnx;
  (void)prim->AddAttr(ops::kFmkType, api::MakeValue(fmk_type));
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxLstmParser("LSTM", new OnnxLstmParser());
}  // namespace lite
}  // namespace mindspore
