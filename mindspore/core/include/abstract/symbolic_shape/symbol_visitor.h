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
#ifndef MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_VISITOR_H_
#define MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_VISITOR_H_
#include "abstract/symbolic_shape/symbol.h"
#include "abstract/symbolic_shape/operation.h"

namespace mindspore {
namespace symshape {
class MS_CORE_API SymbolVisitor {
 public:
  SymbolVisitor() = default;
  virtual ~SymbolVisitor() = default;

  void Visit(Symbol *symbol);
  virtual void VisitImpl(DynamicSymbol *symbol) {}
  virtual void VisitImpl(ScalarSymbol *symbol) {}
  virtual void VisitImpl(IntSymbol *symbol) { VisitImpl(static_cast<ScalarSymbol *>(symbol)); }
  virtual void VisitImpl(BoolSymbol *symbol) { VisitImpl(static_cast<ScalarSymbol *>(symbol)); }
  virtual void VisitImpl(FloatSymbol *symbol) { VisitImpl(static_cast<ScalarSymbol *>(symbol)); }
  virtual void VisitImpl(StrSymbol *symbol) { VisitImpl(static_cast<ScalarSymbol *>(symbol)); }
  virtual void VisitImpl(ListSymbol *symbol) {}

  inline void Visit(Operation *op) { return VisitImpl(op); }
  inline void VisitInputs(Operation *op) {
    for (auto &s : op->inputs()) {
      Visit(s.get());
    }
  }
  virtual void VisitImpl(Operation *op) { VisitInputs(op); }
};
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_VISITOR_H_
