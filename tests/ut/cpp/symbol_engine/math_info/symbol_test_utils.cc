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

#include "symbol_engine/math_info/symbol_test_utils.h"

namespace mindspore::symshape::test {
SymbolPtr SymbolHelper::Emit(const OpPtr &op) {
  InitSymbolEngine();
  return symbol_engine_->emitter_->Emit(op);
}

void SymbolHelper::InitSymbolEngine() {
  if (symbol_engine_ != nullptr) {
    return;
  }
  symbol_engine_ = std::make_shared<InnerSymbolEngine>();
  symbol_engine_->emitter_ = std::make_unique<OperationEmitter>(&symbol_engine_->ops_);
}
}  // namespace mindspore::symshape::test
