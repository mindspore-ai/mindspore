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
#include "mindspore/core/symbolic_shape/operation_builder.h"
#include "mindspore/core/ops/symbol_ops_impl/elemwise_binop.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr Process(OperationBuilder *b, const SymbolPtrList &symbols) {
  if (symbols.empty()) {
    return nullptr;
  }
  auto result = symbols[0];
  for (size_t i = 1; i < symbols.size(); i++) {
    result = b->Emit(std::make_shared<ElemwiseBinop>(result, symbols[i]));
  }
  return result;
}

REG_SYMBOL_OP_BUILDER("AddN").SetShapeFunc([](OperationBuilder *b) {
  // inputs are spread
  // todo, remove this branch
  if (b->input_num() > kDim1) {
    SymbolPtrList symbols(b->input_num());
    for (size_t i = 0; i < symbols.size(); i++) {
      symbols[i] = b->GetInputShape(i);
    }
    return Process(b, symbols);
  }

  auto inputs = b->GetInputShape(kIndex0)->as_sptr<ListSymbol>();
  MS_EXCEPTION_IF_NULL(inputs);
  return Process(b, inputs->symbols());
});
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
