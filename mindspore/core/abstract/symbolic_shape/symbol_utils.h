/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_UTILS_H_
#define MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_UTILS_H_

#include <vector>
#include <string>
#include "mindapi/base/shape_vector.h"
#include "include/abstract/symbolic_shape/symbol.h"

namespace mindspore {
namespace symshape {
class Symbol;
using SymbolPtr = std::shared_ptr<Symbol>;

SymbolPtr ShapeVector2Symbol(const ShapeVector &shape, const OpPtr &op = nullptr);

std::string SymbolListToStr(const SymbolPtrList &slist, const std::string &pre, const std::string &post,
                            bool raw_str = false);
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_UTILS_H_
