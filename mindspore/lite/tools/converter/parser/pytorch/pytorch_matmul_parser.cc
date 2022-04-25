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

#include "tools/converter/parser/pytorch/pytorch_matmul_parser.h"
#include <memory>
#include "ops/fusion/mat_mul_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr PytorchAddmmParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::MatMulFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  input_indices->resize(kInputSize2);
  std::iota(input_indices->begin(), input_indices->end(), 1);
  input_indices->at(THIRD_INPUT) = 0;

  int64_t alpha = PytorchNodeParser::GetValueFromConstNode<int64_t>(torch_node->input(FIFTH_INPUT));
  int64_t beta = PytorchNodeParser::GetValueFromConstNode<int64_t>(torch_node->input(FOURTH_INPUT));
  if (alpha != 1 || beta != 1) {
    MS_LOG(ERROR) << "not support alpha * A * B + beta * C";
    return nullptr;
  }

  prim->set_activation_type(mindspore::NO_ACTIVATION);
  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchMatmulParser("addmm", new PytorchAddmmParser());
}  // namespace lite
}  // namespace mindspore
