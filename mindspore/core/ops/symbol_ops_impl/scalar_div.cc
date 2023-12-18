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
#include "mindspore/core/ops/symbol_ops_impl/scalar_div.h"
#include <algorithm>
#include <vector>

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr ScalarDiv::Eval() {
  // only eval on Building
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(lhs->value() / rhs->value());
  }
  if (lhs->HasData() && lhs->value() == 0) {
    return GenInt(0);
  }
  if (rhs->HasData() && rhs->value() == 1) {
    DoNotEvalOnRun();
    return input(0);
  }
  return GenVInt();
}

void ScalarDiv::UpdateMathInfo() {
  if (!need_eval()) {
    return;
  }
  auto input1 = input_as_sptr<IntSymbol>(0);
  auto input2 = input_as_sptr<IntSymbol>(1);
  auto out = output_as<IntSymbol>();
  int64_t min1 = input1->range_min();
  int64_t max1 = input1->range_max();
  int64_t min2 = input2->range_min();
  int64_t max2 = input2->range_max();
  std::vector<int64_t> v;
  v.push_back(RangeDiv(min1, min2));
  v.push_back(RangeDiv(min1, max2));
  v.push_back(RangeDiv(max1, min2));
  v.push_back(RangeDiv(max1, max2));
  if (min2 <= -1 && -1 <= max2) {
    v.push_back(-min1);
    v.push_back(-max1);
  }
  if (min2 <= 1 && 1 <= max2) {
    v.push_back(min1);
    v.push_back(max1);
  }
  out->SetRange(*std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
  // only support "s / const", does not support "const / s".
  if (input2->is_const() && !input1->is_const()) {
    // out = input1 / const2
    out->SetMathExpr(input1, Frac(1, input2->value()), 0);
  }
}

REG_SYMBOL_OP_BUILDER("ScalarDiv").SetValueFunc(DefaultBuilder<ScalarDiv, 2>);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
