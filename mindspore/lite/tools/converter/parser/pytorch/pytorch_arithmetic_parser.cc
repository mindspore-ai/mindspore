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

#include "tools/converter/parser/pytorch/pytorch_arithmetic_parser.h"
#include <memory>
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr PytorchAddParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::AddFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  input_indices->push_back(0);
  input_indices->push_back(1);
  int64_t alpha = PytorchNodeParser::GetValueFromConstNode<int64_t>(torch_node->input(THIRD_INPUT));
  if (alpha != 1) {
    MS_LOG(ERROR) << "not support alpha * A + B";
    return nullptr;
  }
  return prim->GetPrim();
}

PrimitiveCPtr PytorchSubParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::SubFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr PytorchMulParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::MulFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr PytorchDivParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::DivFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchAddParser("add", new PytorchAddParser());
PytorchNodeRegistrar g_pytorchSubParser("sub", new PytorchSubParser());
PytorchNodeRegistrar g_pytorchMulParser("mul", new PytorchMulParser());
PytorchNodeRegistrar g_pytorchDivParser("div", new PytorchDivParser());
}  // namespace lite
}  // namespace mindspore
