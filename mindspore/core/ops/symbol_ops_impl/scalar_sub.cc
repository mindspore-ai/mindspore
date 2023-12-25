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
#include "mindspore/core/ops/symbol_ops_impl/scalar_sub.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr ScalarSub::Eval() {
  // only eval on Building
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(lhs->value() - rhs->value());
  }
  if (rhs->HasData() && rhs->value() == 0) {
    DoNotEvalOnRun();
    return input(0);
  }
  return GenVInt();
}
void ScalarSub::UpdateMathInfo() {
  if (!need_eval()) {
    return;
  }
  auto a = input_as_sptr<IntSymbol>(0);
  auto b = input_as_sptr<IntSymbol>(1);
  auto out = output_as<IntSymbol>();
  out->SetRange(RangeSub(a->range_min(), b->range_max()), RangeSub(a->range_max(), b->range_min()));
  if (a->is_const() && !b->is_const()) {
    // out = const_a - b
    out->SetMathExpr(b, Frac(-1), a->value());
  } else if (b->is_const() && !a->is_const()) {
    // out = a - const_b
    out->SetMathExpr(a, kFrac1, -b->value());
  }
}

REG_SYMBOL_OP_BUILDER("ScalarSub").SetValueFunc(DefaultBuilder<ScalarSub, 2>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
