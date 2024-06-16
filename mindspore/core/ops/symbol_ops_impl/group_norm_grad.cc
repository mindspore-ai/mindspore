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

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API GroupNormGrad : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  explicit GroupNormGrad(const SymbolPtr &x) : InferShapeOp({x}) {}
  ~GroupNormGrad() override = default;
  MS_DECLARE_PARENT(GroupNormGrad, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr GroupNormGrad::Eval() {
  auto x = input_as_sptr<ListSymbol>(kIndex0);
  if (!x->HasData()) {
    auto out_shape = GenList({GenVInt()});
    return GenList({x, out_shape, out_shape});
  }
  DoNotEvalOnRun();
  auto out_shape = GenList({x->item(kIndex1)});
  return GenList({x, out_shape, out_shape});
}

REG_SYMBOL_OP_BUILDER("GroupNormGrad")
  .SetShapeDepend({DependOn::kNone, DependOn::kShape})
  .SetShapeFuncWith<GroupNormGrad>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
