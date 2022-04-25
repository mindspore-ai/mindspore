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

#include "tools/converter/parser/pytorch/pytorch_batchnorm_parser.h"
#include <memory>
#include "ops/fused_batch_norm.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kBatchNormInputSize = 5;
constexpr size_t kBatchNormMonmentumIndex = 6;
constexpr size_t kBatchNormEpsilonIndex = 7;
}  // namespace
PrimitiveCPtr PytorchBatchNormParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::FusedBatchNorm>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  input_indices->resize(kBatchNormInputSize);
  std::iota(input_indices->begin(), input_indices->end(), 0);
  auto momentum = PytorchNodeParser::GetValueFromConstNode<float>(torch_node->input(kBatchNormMonmentumIndex));
  auto epsilon = PytorchNodeParser::GetValueFromConstNode<float>(torch_node->input(kBatchNormEpsilonIndex));
  prim->set_momentum(momentum);
  prim->set_epsilon(epsilon);
  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchBatchNormParser("batch_norm", new PytorchBatchNormParser());
}  // namespace lite
}  // namespace mindspore
