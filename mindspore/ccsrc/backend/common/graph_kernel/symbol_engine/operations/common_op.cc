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
#include "backend/common/graph_kernel/symbol_engine/operations/common_op.h"
#include <functional>
#include <utility>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/hash_set.h"

namespace mindspore::graphkernel::symbol {
namespace ops {
SymbolPtr ScalarAdd::Eval() {
  // only eval on Building
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(lhs->value() + rhs->value());
  }
  if (lhs->HasData() && lhs->value() == 0) {
    DoNotEvalOnRun();
    return input(1);
  }
  if (rhs->HasData() && rhs->value() == 0) {
    DoNotEvalOnRun();
    return input(0);
  }
  return GenVInt();
}
void ScalarAdd::UpdateMathInfo() {
  if (!need_eval()) {
    return;
  }
  auto a = input_as_sptr<IntSymbol>(0);
  auto b = input_as_sptr<IntSymbol>(1);
  auto out = output_as<IntSymbol>();
  out->SetRange(RangeAdd(a->range_min(), b->range_min()), RangeAdd(a->range_max(), b->range_max()));
  if (a->is_const() && !b->is_const()) {
    out->SetMathExpr(b, kFrac1, a->value());
  } else if (b->is_const() && !a->is_const()) {
    out->SetMathExpr(a, kFrac1, b->value());
  }
}

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

SymbolPtr ScalarMul::Eval() {
  // only eval on Building
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(lhs->value() * rhs->value());
  }
  if ((lhs->HasData() && lhs->value() == 0) || (rhs->HasData() && rhs->value() == 0)) {
    return GenInt(0);
  }
  if (lhs->HasData() && lhs->value() == 1) {
    DoNotEvalOnRun();
    return input(1);
  }
  if (rhs->HasData() && rhs->value() == 1) {
    DoNotEvalOnRun();
    return input(0);
  }
  return GenVInt();
}
void ScalarMul::UpdateMathInfo() {
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
  int64_t a = RangeMul(min1, min2);
  int64_t b = RangeMul(min1, max2);
  int64_t c = RangeMul(max1, min2);
  int64_t d = RangeMul(max1, max2);
  out->SetRange(std::min({a, b, c, d}), std::max({a, b, c, d}));
  if (input1->is_const() && !input2->is_const()) {
    // out = const1 * input2
    out->SetMathExpr(input2, Frac(input1->value()), 0);
  } else if (input2->is_const() && !input1->is_const()) {
    // out = input1 * const2
    out->SetMathExpr(input1, Frac(input2->value()), 0);
  }
}

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

SymbolPtr ScalarMax::Eval() {
  // only eval on Building
  auto lhs = input_as_sptr<IntSymbol>(0);
  auto rhs = input_as_sptr<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(std::max(lhs->value(), rhs->value()));
  }
  if (*lhs >= *rhs) {
    DoNotEvalOnRun();
    return lhs;
  }
  if (*rhs > *lhs) {
    DoNotEvalOnRun();
    return rhs;
  }
  return GenVInt();
}
void ScalarMax::UpdateMathInfo() {
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  output_as<IntSymbol>()->SetRange(std::max(lhs->range_min(), rhs->range_min()),
                                   std::max(lhs->range_max(), rhs->range_max()));
}

SymbolPtr ScalarMin::Eval() {
  // only eval on Building
  auto lhs = input_as_sptr<IntSymbol>(0);
  auto rhs = input_as_sptr<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(std::min(lhs->value(), rhs->value()));
  }
  if (*lhs <= *rhs) {
    DoNotEvalOnRun();
    return lhs;
  }
  if (*rhs < *lhs) {
    DoNotEvalOnRun();
    return rhs;
  }
  return GenVInt();
}
void ScalarMin::UpdateMathInfo() {
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  output_as<IntSymbol>()->SetRange(std::min(lhs->range_min(), rhs->range_min()),
                                   std::min(lhs->range_max(), rhs->range_max()));
}

SymbolPtr ScalarEQ::Eval() {
  // only eval on Building
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return BoolSymbol::Make(lhs->value() == rhs->value());
  }
  return (*lhs == *rhs) ? BoolSymbol::Make(true) : BoolSymbol::Make(shared_from_this());
}
}  // namespace ops
}  // namespace mindspore::graphkernel::symbol
