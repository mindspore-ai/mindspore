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
#include "tools/converter/parser/onnx/onnx_swin_transformer_ln_qkv_parser.h"
#include "tools/converter/ops/ops_def.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kEpsilon = "epsilon";
constexpr auto kHeadDim = "head_dim";
constexpr auto kHeadNum = "head_num";
constexpr auto kSeqLength = "seq_length";
constexpr auto kShifts = "shifts";
}  // namespace
PrimitiveCPtr OnnxSwinTransformerLnQKVParser::Parse(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<SwinTransformerLnQKV>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == kEpsilon) {
      prim->AddAttr(kEpsilon, MakeValue(static_cast<float>(onnx_node_attr.f())));
    } else if (attribute_name == kHeadDim) {
      prim->AddAttr(kHeadDim, MakeValue(static_cast<int64_t>(onnx_node_attr.i())));
    } else if (attribute_name == kHeadNum) {
      prim->AddAttr(kHeadNum, MakeValue(static_cast<int64_t>(onnx_node_attr.i())));
    } else if (attribute_name == kSeqLength) {
      prim->AddAttr(kSeqLength, MakeValue(static_cast<int64_t>(onnx_node_attr.i())));
    } else if (attribute_name == kShifts) {
      std::vector<int64_t> shifts = {static_cast<int64_t>(onnx_node_attr.i())};
      prim->AddAttr(kShifts, MakeValue(shifts));
    }
  }
  return prim;
}

OnnxNodeRegistrar g_onnxSwinTransformerLnQKVParser("SwinTransformerLnQKV", new OnnxSwinTransformerLnQKVParser());
}  // namespace lite
}  // namespace mindspore
