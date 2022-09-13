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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_ELEMENT_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_ELEMENT_PARSER_H_

#include <vector>
#include "tools/converter/parser/pytorch/pytorch_node_parser.h"
#include "tools/converter/parser/pytorch/pytorch_node_parser_registry.h"

namespace mindspore {
namespace lite {
template <typename OPTy>
class PytorchElementOpParser : public PytorchNodeParser {
 public:
  PytorchElementOpParser() : PytorchNodeParser("ElementWiseOp") {}
  ~PytorchElementOpParser() override = default;

  PrimitiveCPtr Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_ELEMENT_PARSER_H_
