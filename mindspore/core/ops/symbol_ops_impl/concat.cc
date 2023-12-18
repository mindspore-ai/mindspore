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

namespace mindspore {
namespace symshape {
namespace ops {
// only support IntList value for shape calculation
SymbolPtr ConcatValue(OperationBuilder *b) {
  SymbolPtrList result;
  if (b->input_num() == kDim2) {
    // inputs of Concat is a tuple.
    auto inputs = b->GetInputValue(kIndex0)->as_sptr<ListSymbol>();
    if (inputs == nullptr) {
      return nullptr;
    }
    result.reserve(inputs->size());
    for (auto &inp : inputs->symbols()) {
      if (auto ilist = inp->as<ListSymbol>(); ilist != nullptr) {
        (void)result.insert(result.end(), ilist->symbols().begin(), ilist->symbols().end());
      } else if (inp->is<IntSymbol>()) {
        (void)result.emplace_back(inp);
      } else {
        return nullptr;
      }
    }
  } else {
    // inputs of Concat are spread, and the last input is "axis".
    // todo, remove this branch
    result.reserve(b->input_num());
    for (size_t i = 0; i + 1 < b->input_num(); i++) {
      auto v = b->GetInputValue(i)->as_sptr<ListSymbol>();
      if (v != nullptr) {
        (void)result.insert(result.end(), v->symbols().begin(), v->symbols().end());
      } else {
        return nullptr;
      }
    }
  }
  return ListSymbol::Make(std::move(result));
}

REG_SYMBOL_OP_BUILDER("Concat").SetValueFunc(ConcatValue);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
