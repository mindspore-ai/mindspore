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
#include <algorithm>
#include "backend/common/graph_kernel/symbol_engine/utils.h"

namespace mindspore::graphkernel::symbol {
ShapeVector ToShape(const Symbol *symbol) {
  if (!symbol->HasData()) {
    return {abstract::Shape::kShapeRankAny};
  }
  auto *list = symbol->as<ListSymbol>();
  ShapeVector shape(list->size());
  (void)std::transform(list->symbols().cbegin(), list->symbols().cend(), shape.begin(), [](const SymbolPtr &s) {
    auto int_smbl = s->as<IntSymbol>();
    MS_EXCEPTION_IF_NULL(int_smbl);
    if (!int_smbl->HasData()) {
      return abstract::Shape::kShapeDimAny;
    }
    return int_smbl->value();
  });
  return shape;
}
}  // namespace mindspore::graphkernel::symbol
