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

#include "tools/converter/parser/pytorch/pytorch_lstm_parser.h"
#include <memory>
#include "ops/lstm.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kLstmInputsNum = 9;
}

PrimitiveCPtr PytorchLstmParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::LSTM>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  if (torch_node->inputs().size() < kLstmInputsNum) {
    MS_LOG(ERROR) << "lstm input num is less than expected num 9, current num is " << torch_node->inputs().size();
    return nullptr;
  }

  auto has_bias = PytorchNodeParser::GetValueFromConstNode<bool>(torch_node->input(FOURTH_INPUT));
  prim->set_has_bias(has_bias);
  auto num_layers = PytorchNodeParser::GetValueFromConstNode<int64_t>(torch_node->input(FIFTH_INPUT));
  prim->set_num_layers(num_layers);
  auto dropout = PytorchNodeParser::GetValueFromConstNode<float>(torch_node->input(SIXTH_INPUT));
  prim->set_dropout(dropout);

  // SEVENTH_INPUT is train mode (bool)
  auto bidirectional = PytorchNodeParser::GetValueFromConstNode<bool>(torch_node->input(EIGHTH_INPUT));
  prim->set_bidirectional(bidirectional);
  auto batch_first = PytorchNodeParser::GetValueFromConstNode<bool>(torch_node->input(NINTH_INPUT));
  prim->set_has_bias(batch_first);

  input_indices->push_back(FIRST_INPUT);
  input_indices->push_back(SECOND_INPUT);
  input_indices->push_back(THIRD_INPUT);

  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchLstmParser("lstm", new PytorchLstmParser());
}  // namespace lite
}  // namespace mindspore
