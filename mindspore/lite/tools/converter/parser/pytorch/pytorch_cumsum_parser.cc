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

#include "tools/converter/parser/pytorch/pytorch_cumsum_parser.h"
#include <memory>
#include "ops/cumsum.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr PytorchCumSumParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::CumSum>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  input_indices->push_back(0);

  if (torch_node->inputs().size() > SECOND_INPUT) {
    auto dim = PytorchNodeParser::GetValueFromConstNode<int64_t>(torch_node->input(SECOND_INPUT));
    prim->set_axis(dim);
  }

  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchCunSumParser("cumsum", new PytorchCumSumParser());
}  // namespace lite
}  // namespace mindspore
