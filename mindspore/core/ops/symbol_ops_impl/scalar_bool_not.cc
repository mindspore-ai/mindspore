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
#include "mindspore/core/ops/symbol_ops_impl/scalar_cmp.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API ScalarBoolNot final : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  explicit ScalarBoolNot(const SymbolPtr &x) : InferValueOp({x}) {}
  ~ScalarBoolNot() override = default;
  MS_DECLARE_PARENT(ScalarBoolNot, InferValueOp)
 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<BoolSymbol>()->SetValue(!input_as<BoolSymbol>(0)->value()); }
};

SymbolPtr ScalarBoolNot::Eval() {
  auto x = input_as<BoolSymbol>(0);
  if (x->HasData()) {
    return BoolSymbol::Make(!x->value());
  }
  return BoolSymbol::Make(shared_from_this());
}

REG_SYMBOL_OP_BUILDER("BoolNot").SetValueDepend({DependOn::kValue}).SetValueFuncWith<ScalarBoolNot>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
