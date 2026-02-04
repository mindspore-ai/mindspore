/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "mindspore/core/abstract/symbolic_shape/symbol_utils.h"

#include <sstream>
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "abstract/symbolic_shape/symbol.h"
#include "abstract/symbolic_shape/int_symbol.h"

namespace mindspore {
namespace symshape {
SymbolPtr ShapeVector2Symbol(const ShapeVector &shape, const OpPtr &op) {
  if (IsDynamicRank(shape)) {
    return ListSymbol::Make(op);
  }
  SymbolPtrList result(shape.size());
  (void)std::transform(shape.begin(), shape.end(), result.begin(), [op](int64_t s) {
    if (s == abstract::Shape::kShapeDimAny) {
      return IntSymbol::Make(op);
    } else {
      return IntSymbol::Make(s, op);
    }
  });
  return ListSymbol::Make(std::move(result), op);
}

std::string SymbolListToStr(const SymbolPtrList &slist, const std::string &pre, const std::string &post, bool raw_str) {
  std::ostringstream oss;
  oss << pre;
  bool first = true;
  for (auto &s : slist) {
    if (first) {
      first = false;
    } else {
      oss << ", ";
    }
    MS_EXCEPTION_IF_NULL(s);
    oss << (raw_str ? s->ToRawString() : s->ToString());
  }
  oss << post;
  return oss.str();
}
}  // namespace symshape
}  // namespace mindspore
