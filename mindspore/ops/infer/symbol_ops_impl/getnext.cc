/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <map>
#include <algorithm>
#include <utility>
#include "abstract/symbolic_shape/operation_builder.h"
#include "abstract/symbolic_shape/symbol_info.h"

namespace mindspore {
namespace symshape {
namespace ops {
std::vector<SymbolInfoList> ParseSymbolInfo(const ValuePtr &attr) {
  auto inputs = attr->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(inputs);
  std::vector<SymbolInfoList> result(inputs->size());
  size_t i = 0;
  for (auto &tensor : inputs->value()) {
    if (!tensor->isa<ValueSequence>()) {
      continue;
    }
    auto shape = tensor->cast_ptr<ValueSequence>();
    auto &result_i = result[i++];
    result_i.reserve(shape->size());
    for (auto &item : shape->value()) {
      auto &info = result_i.emplace_back(SymbolInfo{});
      if (!item->isa<ValueDictionary>()) {
        continue;
      }
      for (auto &iter : item->cast_ptr<ValueDictionary>()->value()) {
        auto cfg_key = GetValue<std::string>(iter.first);
        if (cfg_key == "max") {
          info.max = GetValue<int64_t>(iter.second);
        }
        if (cfg_key == "min") {
          info.min = GetValue<int64_t>(iter.second);
        }
        if (cfg_key == "divisor") {
          info.divisor = GetValue<int64_t>(iter.second);
        }
        if (cfg_key == "remainder") {
          info.remainder = GetValue<int64_t>(iter.second);
        }
        if (cfg_key == "id") {
          info.id = GetValue<int64_t>(iter.second);
        }
        if (cfg_key == "name") {
          info.name = GetValue<std::string>(iter.second);
        }
      }
    }
  }
  return result;
}

REG_SYMBOL_OP_BUILDER("GetNext").SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
  ValuePtr symbols_attr = nullptr;
  if (b->prim()->HasAttr("symbols_for_parallel")) {
    symbols_attr = b->prim()->GetAttr("symbols_for_parallel");
  } else if (b->prim()->HasAttr("symbols")) {
    symbols_attr = b->prim()->GetAttr("symbols");
  } else {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(symbols_attr);
  MS_EXCEPTION_IF_NULL(b->out_abstract());
  auto abs_seq = b->out_abstract()->cast_ptr<abstract::AbstractSequence>();
  MS_EXCEPTION_IF_NULL(abs_seq);
  auto out = BuildSymbolicShapeBySymbolInfo(abs_seq->elements(), ParseSymbolInfo(symbols_attr));
  if (out.empty()) {
    return nullptr;
  }
  return ListSymbol::Make(SymbolPtrList(out.begin(), out.end()));
});
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
