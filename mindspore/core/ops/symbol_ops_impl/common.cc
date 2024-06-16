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
#include "mindspore/core/ops/symbol_ops_impl/common.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_add.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_mul.h"

namespace mindspore {
namespace symshape {
namespace ops {
void InferShapeOp::SetPositive(const ListSymbol *list) {
  for (auto &s : list->symbols()) {
    auto list_s = s->as_noexcept<ListSymbol>();
    if (list_s != nullptr) {
      SetPositive(list_s);
    } else {
      auto int_s = s->as_noexcept<IntSymbol>();
      MS_EXCEPTION_IF_NULL(int_s);
      if (!int_s->is_positive()) {
        int_s->SetRangeMin(1);
      }
    }
  }
}

SymbolPtr TransValueToShape(OperationBuilder *b) {
  auto ret = TransparentInput(b);
  if (ret == nullptr) {
    return nullptr;
  }
  auto ret_shape = ret->as_noexcept<ListSymbol>();
  MS_EXCEPTION_IF_NULL(ret_shape);
  InferShapeOp::SetPositive(ret_shape);
  return ret;
}

template <typename OP>
SymbolPtr Accumulate(const SymbolPtrList &symbols, const OperationEmitter &e) {
  SymbolPtr vars = nullptr;
  int64_t constv = std::is_same_v<OP, ScalarAdd> ? 0 : 1;
  for (size_t i = 0; i < symbols.size(); i++) {
    auto s = symbols[i]->as_sptr<IntSymbol>();
    MS_EXCEPTION_IF_NULL(s);
    if (s->HasData()) {
      if (std::is_same_v<OP, ScalarAdd>) {
        constv += s->value();
      } else {
        constv *= s->value();
      }
    } else if (vars == nullptr) {
      vars = s;
    } else {
      vars = e.Emit(std::make_shared<OP>(vars, s));
    }
  }
  if (vars == nullptr) {
    return IntSymbol::Make(constv);
  }
  return e.Emit(std::make_shared<OP>(vars, IntSymbol::Make(constv)));
}
template SymbolPtr Accumulate<ScalarAdd>(const SymbolPtrList &, const OperationEmitter &);
template SymbolPtr Accumulate<ScalarMul>(const SymbolPtrList &, const OperationEmitter &);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
