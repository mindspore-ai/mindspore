/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/pytorch/pytorch_to_parser.h"
#include "tools/converter/parser/pytorch/pytorch_node_parser.h"
#include <memory>
#include "ops/cast.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr PytorchToParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::Cast>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  input_indices->push_back(0);

  auto pytorch_dtype = PytorchNodeParser::GetValueFromConstNode<int8_t>(torch_node->input(SECOND_INPUT));
  auto dst_type = GetDataTypeFromTorch(static_cast<at::ScalarType>(pytorch_dtype));
  if (dst_type == kNumberTypeInt64) {
    dst_type = kNumberTypeInt32;
  }
  if (dst_type == kNumberTypeFloat64) {
    dst_type = kNumberTypeFloat32;
  }
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  prim_c->AddAttr("to", MakeValue(static_cast<int32_t>(dst_type)));

  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchToParser("to", new PytorchToParser());
}  // namespace lite
}  // namespace mindspore
