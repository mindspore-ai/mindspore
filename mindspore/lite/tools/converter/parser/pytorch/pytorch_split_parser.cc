/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/pytorch/pytorch_split_parser.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "ops/split.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr PytorchSplitParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  auto prim = std::make_unique<ops::Split>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  input_indices->push_back(0);

  int64_t split_num = 0;
  if (torch_node->inputs().size() > SECOND_INPUT) {
    auto size_splits = PytorchNodeParser::GetValueFromConstNode<std::vector<int64_t>>(torch_node->input(SECOND_INPUT));
    prim->set_size_splits(size_splits);
    split_num = size_splits.size();
  }
  if (torch_node->inputs().size() > THIRD_INPUT) {
    auto dim = PytorchNodeParser::GetValueFromConstNode<int64_t>(torch_node->input(THIRD_INPUT));
    prim->set_axis(dim);
  }

  if (split_num == 0) {
    split_num = torch_node->outputs().size();
  }
  prim->set_output_num(split_num);
  return prim->GetPrim();
}
PytorchNodeRegistrar g_pytorchSplitParser("split_with_sizes", new PytorchSplitParser());
}  // namespace lite
}  // namespace mindspore
