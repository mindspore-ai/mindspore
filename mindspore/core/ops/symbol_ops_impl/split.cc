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
#include "mindspore/core/ops/symbol_ops_impl/common.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API Split : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Split(const SymbolPtr &x, const SymbolPtr &axis, const SymbolPtr &out_num) : InferShapeOp({x, axis, out_num}) {}
  ~Split() override = default;
  MS_DECLARE_PARENT(Split, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr Split::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  auto axis = input_as<IntSymbol>(kIndex1);
  auto out_num = input_as_sptr<IntSymbol>(kIndex2);
  auto out_num_v = LongToSize(out_num->value());  // only support the output_num is a const value
  if (!x->HasData() || !axis->HasData()) {
    // all output shapes are equal
    return GenList(SymbolPtrList(out_num_v, GenVList()));
  }
  DoNotEvalOnRun();
  auto axis_v = LongToSize(NormAxis(axis->value(), x->size()));
  SymbolPtrList out_shape = x->symbols();
  out_shape[axis_v] = Emit(std::make_shared<ScalarDiv>(x->item(axis_v), out_num));
  return GenList(SymbolPtrList(out_num_v, GenList(std::move(out_shape))));
}

REG_SYMBOL_OP_BUILDER("Split")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue})
  .SetShapeFuncWith<Split>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
