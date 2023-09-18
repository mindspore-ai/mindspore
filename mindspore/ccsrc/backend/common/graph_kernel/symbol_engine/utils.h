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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_UTILS_H_

#include "backend/common/graph_kernel/symbol_engine/symbol.h"

namespace mindspore::graphkernel::symbol {
// symbol to ShapeVector
ShapeVector ToShape(const Symbol *symbol);
inline ShapeVector ToShape(const SymbolPtr &symbol) { return ToShape(symbol.get()); }

// get int value from symbol
inline int64_t AsInt(const Symbol *s) {
  auto v = s->as<IntSymbol>();
  MS_EXCEPTION_IF_NULL(v);
  return v->value();
}
inline int64_t AsInt(const SymbolPtr &s) { return AsInt(s.get()); }

// get int value from symbol
inline bool AsBool(const Symbol *s) {
  auto v = s->as<BoolSymbol>();
  MS_EXCEPTION_IF_NULL(v);
  return v->value();
}
inline bool AsBool(const SymbolPtr &s) { return AsBool(s.get()); }

// calculations of the Range value
inline int64_t Sign(int64_t x) { return x < 0 ? -1 : 1; }
inline int64_t GenInf(int64_t x) { return Sign(x) * kINF; }
inline int64_t RangeAdd(int64_t a, int64_t b) { return std::abs(a) == kINF ? a : std::abs(b) == kINF ? b : a + b; }
inline int64_t RangeSub(int64_t a, int64_t b) { return RangeAdd(a, -b); }
inline int64_t RangeMul(int64_t a, int64_t b) {
  return (std::abs(a) == kINF || std::abs(b) == kINF) ? GenInf(a * b) : a * b;
}
inline int64_t RangeDiv(int64_t a, int64_t b) {
  if (b == 0) {
    return GenInf(a);
  }
  if (std::abs(b) == kINF) {
    return 0;
  }
  return std::abs(a) == kINF ? GenInf(a * b) : a / b;
}
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_UTILS_H_
