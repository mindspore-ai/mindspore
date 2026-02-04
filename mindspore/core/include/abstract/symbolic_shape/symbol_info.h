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
#ifndef MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_INFO_H_
#define MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_INFO_H_

#include <vector>
#include <string>
#include <optional>
#include "abstract/abstract_value.h"

namespace mindspore {
namespace symshape {
struct MS_CORE_API SymbolInfo {
  int64_t max = -1;
  int64_t min = 1;
  int64_t divisor = 1;
  int64_t remainder = 0;
  std::optional<int64_t> id = std::nullopt;
  std::string name;
};
using SymbolInfoList = std::vector<SymbolInfo>;

MS_CORE_API std::vector<ListSymbolPtr> BuildSymbolicShapeBySymbolInfo(const AbstractBasePtrList &args_abs,
                                                                      const std::vector<SymbolInfoList> &symbol_infos);
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_INFO_H_
