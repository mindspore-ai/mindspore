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

#include "abstract/symbolic_shape/symbol_visitor.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace symshape {
#define SYMBOL_DISPATCH(cls)            \
  case cls::kTypeId:                    \
    VisitImpl(static_cast<cls *>(ptr)); \
    break

void SymbolVisitor::Visit(Symbol *ptr) {
  switch (ptr->tid()) {
    SYMBOL_DISPATCH(DynamicSymbol);
    SYMBOL_DISPATCH(ListSymbol);
    SYMBOL_DISPATCH(IntSymbol);
    SYMBOL_DISPATCH(BoolSymbol);
    SYMBOL_DISPATCH(FloatSymbol);
    SYMBOL_DISPATCH(StrSymbol);
    default:
      // do not dispatch the 'ScalarSymbol'
      MS_LOG(WARNING) << "This type of symbol " << ptr->type_name() << " is not dispatched.";
  }
}
}  // namespace symshape
}  // namespace mindspore
