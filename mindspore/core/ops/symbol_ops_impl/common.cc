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

namespace mindspore {
namespace symshape {
namespace ops {
void InferShapeOp::SetPositive(const ListSymbol *list) {
  for (auto &s : list->symbols()) {
    auto list_s = s->as<ListSymbol>();
    if (list_s != nullptr) {
      SetPositive(list_s);
    } else {
      auto int_s = s->as<IntSymbol>();
      MS_EXCEPTION_IF_NULL(int_s);
      if (!int_s->is_positive()) {
        int_s->SetRangeMin(1);
      }
    }
  }
}

SymbolPtr TransparentInput(OperationBuilder *b) {
  bool build_value = !b->is_building_shape();
  auto depends = b->symbol_builder_info().GetDepends(b->prim(), build_value);
  if (depends.empty()) {
    (void)depends.emplace_back((build_value ? DependOn::kValue : DependOn::kShape));
  }
  // check only one depend status in the list.
  auto iter1 = std::find_if(depends.begin(), depends.end(), [](DependOn d) { return d != DependOn::kNone; });
  if (iter1 == depends.end()) {
    return nullptr;
  }
  auto iter2 = std::find_if(iter1 + 1, depends.end(), [](DependOn d) { return d != DependOn::kNone; });
  if (iter2 != depends.end()) {
    return nullptr;
  }
  size_t idx = iter1 - depends.begin();
  return (*iter1 == DependOn::kShape) ? b->GetInputShape(idx) : b->GetInputValue(idx);
}
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
