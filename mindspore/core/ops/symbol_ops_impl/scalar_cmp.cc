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
#include "mindspore/core/ops/symbol_ops_impl/scalar_cmp.h"

namespace mindspore {
namespace symshape {
namespace ops {
SymbolPtr ScalarCmpOp::Eval() {
  // only eval on Building
  auto lhs = input_as<IntSymbol>(0);
  auto rhs = input_as<IntSymbol>(1);
  if (lhs->HasData() && rhs->HasData()) {
    return BoolSymbol::Make(Compare(lhs, rhs));
  }
  return Compare(lhs, rhs) ? BoolSymbol::Make(true) : BoolSymbol::Make(shared_from_this());
}

REG_SYMBOL_OP_BUILDER("ScalarEq").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarEq>();
REG_SYMBOL_OP_BUILDER("ScalarGe").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarGe>();
REG_SYMBOL_OP_BUILDER("ScalarGt").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarGt>();
REG_SYMBOL_OP_BUILDER("ScalarLe").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarLe>();
REG_SYMBOL_OP_BUILDER("ScalarLt").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarLt>();
REG_SYMBOL_OP_BUILDER("scalar_eq").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarEq>();
REG_SYMBOL_OP_BUILDER("scalar_ge").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarGe>();
REG_SYMBOL_OP_BUILDER("scalar_gt").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarGt>();
REG_SYMBOL_OP_BUILDER("scalar_le").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarLe>();
REG_SYMBOL_OP_BUILDER("scalar_lt").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarLt>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
