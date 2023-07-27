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

#include "tools/converter/parser/onnx/onnx_gru_parser.h"
#include <memory>
#include <string>
#include <vector>
#include "ops/gru.h"
#include "nnacl/op_base.h"
#include "include/registry/converter_context.h"
#include "mindspore/core/ops/grad/gru_v2_grad.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxGruParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::GRU>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "direction") {
      const auto &direction = onnx_node_attr.s();
      bool bidirectional = direction == "bidirectional";
      prim->set_bidirectional(bidirectional);
    } else if (onnx_node_attr.name() == "activation_alpha") {
      std::vector<float> activation_alpha;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        activation_alpha.push_back(onnx_node_attr.floats(i));
      }
      (void)prim->AddAttr("activation_alpha", api::MakeValue(activation_alpha));
    } else if (onnx_node_attr.name() == "activation_beta") {
      std::vector<float> activation_beta;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        activation_beta.push_back(onnx_node_attr.floats(i));
      }
      (void)prim->AddAttr("activation_beta", api::MakeValue(activation_beta));
    } else if (onnx_node_attr.name() == "activations") {
      std::vector<std::string> activations;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        activations.push_back(onnx_node_attr.strings(i));
      }
      (void)prim->AddAttr("activations", api::MakeValue(activations));
    } else if (onnx_node_attr.name() == "clip") {
      (void)prim->AddAttr("clip", api::MakeValue(onnx_node_attr.f()));
    } else if (onnx_node_attr.name() == "hidden_size") {
      (void)prim->AddAttr("hidden_size", api::MakeValue(onnx_node_attr.i()));
    } else if (onnx_node_attr.name() == "linear_before_reset") {
      (void)prim->AddAttr("linear_before_reset", api::MakeValue(onnx_node_attr.i()));
    }
  }

  int fmk_type = mindspore::converter::FmkType::kFmkTypeOnnx;
  (void)prim->AddAttr(ops::kFmkType, api::MakeValue(fmk_type));
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxGruParser("GRU", new OnnxGruParser());
}  // namespace lite
}  // namespace mindspore
