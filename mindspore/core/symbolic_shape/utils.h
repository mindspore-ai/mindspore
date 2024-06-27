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
#ifndef MINDSPORE_CORE_SYMBOLIC_SHAPE_UTILS_H_
#define MINDSPORE_CORE_SYMBOLIC_SHAPE_UTILS_H_

#include <vector>
#include <string>
#include <set>
#include "mindspore/core/symbolic_shape/symbol.h"
#include "mindspore/core/abstract/abstract_value.h"

namespace mindspore {
namespace symshape {
/// \brief Build constant symbolic value.
MS_CORE_API SymbolPtr ConstValueToSymbol(const ValuePtr &v, bool to_scalar = false);

/// \brief Build symbolic value.
/// If the abstract's value is ValueAny, the variable value list is generated according to the shape.
MS_CORE_API SymbolPtr BuildSymbolicValue(const AbstractBasePtr &abstract);

// symbol to ShapeVector
MS_CORE_API ShapeVector ToShape(const Symbol *symbol);
inline ShapeVector ToShape(const SymbolPtr &symbol) { return ToShape(symbol.get()); }

MS_CORE_API SymbolPtr ShapeVector2Symbol(const ShapeVector &shape, const OpPtr &op = nullptr);

MS_CORE_API SymbolPtr IntValues2Symbol(const std::vector<int64_t> &shape, const OpPtr &op = nullptr);

// get int value from symbol
MS_CORE_API int64_t AsInt(const Symbol *s);
inline int64_t AsInt(const SymbolPtr &s) { return AsInt(s.get()); }

// get bool value from symbol
inline bool AsBool(const Symbol *s) { return s->as<BoolSymbol>()->value(); }
inline bool AsBool(const SymbolPtr &s) { return AsBool(s.get()); }

inline int64_t NormAxis(int64_t axis, size_t rank) { return axis >= 0 ? axis : axis + static_cast<int64_t>(rank); }
MS_CORE_API std::set<int64_t> NormAxis(const ListSymbol *axis, size_t rank);

MS_CORE_API std::string SymbolListToStr(const SymbolPtrList &slist, const std::string &pre, const std::string &post,
                                        bool raw_str = false);

MS_CORE_API BaseShapePtr QueryShape(const AbstractBasePtr &abs);
MS_CORE_API ValuePtr QueryValue(const AbstractBasePtr &abs);
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_SYMBOLIC_SHAPE_UTILS_H_
