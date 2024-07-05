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
#include "mindspore/core/ops/symbol_ops_impl/scalar_mod.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr ScalarMod::Eval() {
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return GenInt(lhs->value() % rhs->value());
  }
  if ((lhs->HasData() && lhs->value() == 0) || (rhs->HasData() && rhs->value() == 1)) {
    return GenInt(0);
  }
  return GenVInt();
}
REG_SYMBOL_OP_BUILDER("ScalarMod").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarMod>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
