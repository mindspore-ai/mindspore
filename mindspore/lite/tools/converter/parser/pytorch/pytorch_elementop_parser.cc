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

#include "tools/converter/parser/pytorch/pytorch_elementop_parser.h"
#include <memory>
#include "ops/less_equal.h"
#include "ops/equal.h"
#include "ops/less.h"
#include "ops/greater.h"
#include "ops/logical_and.h"
#include "ops/logical_not.h"
#include "ops/logical_or.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
template <typename OPTy>
PrimitiveCPtr PytorchElementOpParser<OPTy>::Parse(const torch::jit::Node *torch_node,
                                                  std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<OPTy>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  input_indices->resize(kInputSize1);
  std::iota(input_indices->begin(), input_indices->end(), 0);

  return prim->GetPrim();
}
PytorchNodeRegistrar g_pytorchLeParser("le", new PytorchElementOpParser<ops::LessEqual>());
PytorchNodeRegistrar g_pytorchEqualParser("equal", new PytorchElementOpParser<ops::Equal>());
PytorchNodeRegistrar g_pytorchLessParser("less", new PytorchElementOpParser<ops::Less>());
PytorchNodeRegistrar g_pytorchGreaterParser("greater", new PytorchElementOpParser<ops::Greater>());
PytorchNodeRegistrar g_pytorchGTParser("gt", new PytorchElementOpParser<ops::Greater>());
PytorchNodeRegistrar g_pytorchAndParser("logical_and", new PytorchElementOpParser<ops::LogicalAnd>());
PytorchNodeRegistrar g_pytorchOrParser("logical_or", new PytorchElementOpParser<ops::LogicalOr>());
PytorchNodeRegistrar g_pytorchNotParser("logical_not", new PytorchElementOpParser<ops::LogicalNot>());
}  // namespace lite
}  // namespace mindspore
