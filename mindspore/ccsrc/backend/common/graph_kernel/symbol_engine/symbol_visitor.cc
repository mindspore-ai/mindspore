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

#include "utils/log_adapter.h"
#include "backend/common/graph_kernel/symbol_engine/symbol_visitor.h"

namespace mindspore::graphkernel::symbol {
#define SYMBOL_DISPATCH(cls)        \
  case cls ::kTypeId:               \
    Visit(static_cast<cls *>(ptr)); \
    break;

void SymbolVisitor::Visit(Symbol *ptr) {
  switch (ptr->tid()) {
    SYMBOL_DISPATCH(DynamicSymbol)
    SYMBOL_DISPATCH(InputSymbol)
    SYMBOL_DISPATCH(ScalarSymbol)
    SYMBOL_DISPATCH(IntSymbol)
    SYMBOL_DISPATCH(BoolSymbol)
    SYMBOL_DISPATCH(FloatSymbol)
    SYMBOL_DISPATCH(ListSymbol)
    SYMBOL_DISPATCH(IListSymbol)
    default:
      MS_LOG(WARNING) << "This type of symbol " << ptr->type_name() << " haven't implemented visit ";
  }
}

void SymbolVisitor::Visit(ops::Operation *ptr) {
  switch (ptr->tid()) {
    SYMBOL_DISPATCH(ops::ScalarAdd)
    SYMBOL_DISPATCH(ops::ScalarSub)
    SYMBOL_DISPATCH(ops::ScalarMul)
    SYMBOL_DISPATCH(ops::ScalarDiv)
    SYMBOL_DISPATCH(ops::ScalarMin)
    SYMBOL_DISPATCH(ops::ScalarMax)
    SYMBOL_DISPATCH(ops::Product)
    SYMBOL_DISPATCH(ops::Find)
    SYMBOL_DISPATCH(ops::SetValue)
    SYMBOL_DISPATCH(ops::ListAppend)

    SYMBOL_DISPATCH(ops::infershape::RealShape)
    SYMBOL_DISPATCH(ops::infershape::BinElemwise)
    SYMBOL_DISPATCH(ops::infershape::Reduce)
    SYMBOL_DISPATCH(ops::infershape::Reshape)
    SYMBOL_DISPATCH(ops::infershape::Transpose)
    SYMBOL_DISPATCH(ops::infershape::MatMul)

    SYMBOL_DISPATCH(ops::infervalue::RealValue)
    default:
      MS_LOG(WARNING) << "This type of operation " << ptr->name() << " haven't implemented visit ";
  }
}

}  // namespace mindspore::graphkernel::symbol
