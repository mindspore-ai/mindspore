/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_prompt_flash_attention_parser.h"
#include <memory>
#include <string>
#include <vector>
#include "ops/custom.h"
#include "ops/prompt_flash_attention.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int64_t kNumNextTokens = 65536;
}
PrimitiveCPtr OnnxPromptFlashAttentionParser::Parse(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::PromptFlashAttention>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  int64_t num_heads = 0;
  prim_c->AddAttr("input_layout", MakeValue<std::string>("BNSD"));
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "num_heads") {
      num_heads = onnx_node_attr.i();
      prim_c->AddAttr("num_heads", MakeValue<int64_t>(num_heads));
    } else if (attribute_name == "num_key_value_heads") {
      int64_t num_key_value_heads = onnx_node_attr.i();
      prim_c->AddAttr("num_key_value_heads", MakeValue<int64_t>(num_key_value_heads));
    } else if (attribute_name == "scale_value") {
      auto scale_value = onnx_node_attr.f();
      prim_c->AddAttr("scale_value", MakeValue<float>(scale_value));
    } else if (attribute_name == "next_tokens") {
      int64_t next_token = onnx_node_attr.i();
      prim_c->AddAttr("next_tokens", MakeValue<int64_t>(next_token));
    } else if (attribute_name == "inner_precise") {
      int64_t inner_precise = onnx_node_attr.i();
      prim_c->AddAttr("inner_precise", MakeValue<int64_t>(inner_precise));
    }
  }
  auto next_token_res_ptr = prim_c->GetAttr("next_tokens");
  if (next_token_res_ptr == nullptr) {
    prim_c->AddAttr("next_tokens", MakeValue<int64_t>(kNumNextTokens));
  }

  auto inner_precise_ptr = prim_c->GetAttr("inner_precise");
  if (inner_precise_ptr == nullptr) {
    prim_c->AddAttr("inner_precise", MakeValue<int64_t>(1));
  }

  auto num_key_value_heads_ptr = prim_c->GetAttr("num_key_value_heads");
  if (num_key_value_heads_ptr == nullptr) {
    prim_c->AddAttr("num_key_value_heads", MakeValue<int64_t>(num_heads));
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxPromptFlashAttentionParser("PromptFlashAttention", new OnnxPromptFlashAttentionParser());
}  // namespace lite
}  // namespace mindspore
