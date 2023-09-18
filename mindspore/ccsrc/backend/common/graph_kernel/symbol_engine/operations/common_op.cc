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
  auto a = input_as<IntSymbol>(0);
  auto b = input_as<IntSymbol>(1);
  output_as<IntSymbol>()->SetRange(RangeAdd(a->range_min(), b->range_min()), RangeAdd(a->range_max(), b->range_max()));
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
  auto a = input_as<IntSymbol>(0);
  auto b = input_as<IntSymbol>(1);
  output_as<IntSymbol>()->SetRange(RangeSub(a->range_min(), b->range_max()), RangeSub(a->range_max(), b->range_min()));
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
  auto input1 = input_as<IntSymbol>(0);
  auto input2 = input_as<IntSymbol>(1);
  int64_t min1 = input1->range_min();
  int64_t max1 = input1->range_max();
  int64_t min2 = input2->range_min();
  int64_t max2 = input2->range_max();
  int64_t a = RangeMul(min1, min2);
  int64_t b = RangeMul(min1, max2);
  int64_t c = RangeMul(max1, min2);
  int64_t d = RangeMul(max1, max2);
  output_as<IntSymbol>()->SetRange(std::min({a, b, c, d}), std::max({a, b, c, d}));
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
  auto input1 = input_as<IntSymbol>(0);
  auto input2 = input_as<IntSymbol>(1);
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
  output_as<IntSymbol>()->SetRange(*std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
}

SymbolPtr ScalarMax::Eval() {
  // only eval on Building
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(std::max(lhs->value(), rhs->value()));
  }
  if (*lhs == *rhs) {
    DoNotEvalOnRun();
    return input(0);
  }
  if (lhs->range_min() > rhs->range_max()) {
    DoNotEvalOnRun();
    return input(0);
  }
  if (rhs->range_min() > lhs->range_max()) {
    DoNotEvalOnRun();
    return input(1);
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
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(std::min(lhs->value(), rhs->value()));
  }
  if (*lhs == *rhs) {
    DoNotEvalOnRun();
    return input(0);
  }
  if (lhs->range_max() < rhs->range_min()) {
    DoNotEvalOnRun();
    return input(0);
  }
  if (rhs->range_max() < lhs->range_min()) {
    DoNotEvalOnRun();
    return input(1);
  }
  return GenVInt();
}
void ScalarMin::UpdateMathInfo() {
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  output_as<IntSymbol>()->SetRange(std::min(lhs->range_min(), rhs->range_min()),
                                   std::min(lhs->range_max(), rhs->range_max()));
}

SymbolPtr Product::Eval() {
  auto data = input_as<IListSymbol>(0);
  if (is_building() && !data->AllHaveData()) {
    return GenVInt();
  }
  auto shape = ToShape(data);
  return GenInt(std::accumulate(shape.cbegin(), shape.cend(), 1LL, std::multiplies<int64_t>()));
}

SymbolPtr Find::Eval() {
  auto inp = input_as<ListSymbol>(0);
  auto value = input(1);
  if (!inp->HasData()) {
    return GenVInt();
  }
  auto &list = inp->symbols();
  for (size_t i = 0; i < list.size(); i++) {
    if (list[i]->EqualsTo(value)) {
      return GenInt(SizeToLong(i));
    }
  }
  return GenVInt();
}

SymbolPtr SetValue::Eval() {
  auto inp = input_as<ListSymbol>(kIndex0);
  auto index = input_as<IntSymbol>(kIndex1);
  if (!inp->HasData()) {
    return GenVList();
  }
  if (!index->HasData()) {
    return GenVIntList(inp->size());
  }
  // on Building, if 'input' is static rank and 'index' is const value, unnecessary to evaluate on Run.
  DoNotEvalOnRun();
  auto list = inp->symbols();
  auto idx = index->value();
  if (LongToSize(idx) >= list.size()) {
    MS_LOG(EXCEPTION) << "Index " << idx << " is out of range of list " << inp->ToString();
  }
  list[idx] = input(kIndex2);
  if (is_building()) {
    return GenList(std::move(list));
  }
  output_as<ListSymbol>()->UpdateList(std::move(list));
  return nullptr;
}

SymbolPtr ListAppend::Eval() {
  auto a = input(kIndex0)->as<ListSymbol>();
  MS_EXCEPTION_IF_NULL(a);
  auto b = input(kIndex1);
  auto b_list = b->as<ListSymbol>();
  if (!a->HasData() || (b_list != nullptr && !b_list->HasData())) {
    return GenVList();
  }
  DoNotEvalOnRun();
  SymbolPtrList result;
  result.reserve(a->size() + (b_list != nullptr ? b_list->size() : 1));
  (void)result.insert(result.end(), a->symbols().begin(), a->symbols().end());
  if (b_list != nullptr) {
    (void)result.insert(result.end(), b_list->symbols().begin(), b_list->symbols().end());
  } else {
    (void)result.emplace_back(b);
  }
  if (is_building()) {
    return GenList(std::move(result));
  }
  output_as<ListSymbol>()->UpdateList(std::move(result));
  return nullptr;
}
}  // namespace ops
}  // namespace mindspore::graphkernel::symbol
