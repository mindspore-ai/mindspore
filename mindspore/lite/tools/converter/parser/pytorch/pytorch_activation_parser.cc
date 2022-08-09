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

#include "tools/converter/parser/pytorch/pytorch_activation_parser.h"
#include <memory>
#include "securec/include/securec.h"
#include "ops/fusion/activation.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr PytorchReluParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::RELU);
  prim->set_min_val(0);
  prim->set_max_val(FLT_MAX);

  return prim->GetPrim();
}

PrimitiveCPtr PytorchTanhParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::TANH);

  return prim->GetPrim();
}

PrimitiveCPtr PytorchSigmoidParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::SIGMOID);

  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchReluParser("relu", new PytorchReluParser());
PytorchNodeRegistrar g_pytorchTanhParser("tanh", new PytorchTanhParser());
PytorchNodeRegistrar g_pytorchSigmoidParser("sigmoid", new PytorchSigmoidParser());
}  // namespace lite
}  // namespace mindspore
