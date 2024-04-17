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
#include <map>
#include <string>
#include "tools/converter/parser/onnx/onnx_swin_attention_score_parser.h"
#include "tools/converter/ops/ops_def.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kKeepProb = "keep_prob";
constexpr auto kQueryTranspose = "query_transpose";
constexpr auto kKeyTranspose = "key_transpose";
constexpr auto kBmmScoreTransposeA = "bmm_score_transpose_a";
constexpr auto kBmmScoreTransposeB = "bmm_score_transpose_b";
constexpr auto kSoftmaxAxes = "softmax_axes";

enum AttrDataType { FLOAT, BOOL, LIST_INT };
}  // namespace

PrimitiveCPtr OnnxSwinAttentionScoreParser::Parse(const onnx::GraphProto &onnx_graph,
                                                  const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<SwinAttentionScore>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::map<std::string, AttrDataType> attr_map = {
    {kKeepProb, AttrDataType::FLOAT},          {kQueryTranspose, AttrDataType::BOOL},
    {kKeyTranspose, AttrDataType::BOOL},       {kBmmScoreTransposeA, AttrDataType::BOOL},
    {kBmmScoreTransposeB, AttrDataType::BOOL}, {kSoftmaxAxes, AttrDataType::LIST_INT}};
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    auto attr_data_type = attr_map.find(attribute_name);
    if (attr_data_type != attr_map.end()) {
      std::vector<int64_t> softmax_axes;
      switch (attr_data_type->second) {
        case AttrDataType::FLOAT:
          prim->AddAttr(attr_data_type->first, MakeValue(static_cast<float>(onnx_node_attr.f())));
          break;
        case AttrDataType::BOOL:
          prim->AddAttr(attr_data_type->first, MakeValue(static_cast<bool>(onnx_node_attr.i())));
          break;
        case AttrDataType::LIST_INT:
          softmax_axes.resize(onnx_node_attr.ints_size());
          std::copy(onnx_node_attr.ints().begin(), onnx_node_attr.ints().end(), softmax_axes.begin());
          prim->AddAttr(kSoftmaxAxes, MakeValue(softmax_axes));
          break;
        default:
          MS_LOG(ERROR) << "Unexpected Attributes Data Type[" << attr_data_type->second << "] from "
                        << attr_data_type->first;
          return nullptr;
      }
    }
  }
  return prim;
}

OnnxNodeRegistrar g_onnxSwinAttentionScoreParser("SwinAttentionScore", new OnnxSwinAttentionScoreParser());
}  // namespace lite
}  // namespace mindspore
