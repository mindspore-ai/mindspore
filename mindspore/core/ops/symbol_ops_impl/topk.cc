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
class MS_CORE_API TopK : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  TopK(const SymbolPtr &x, const SymbolPtr &k) : InferShapeOp({x, k}) {}
  ~TopK() override = default;
  MS_DECLARE_PARENT(TopK, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr TopK::Eval() {
  auto x = input_as<ListSymbol>(kIndex0);
  SymbolPtr k = input(kIndex1);
  if (k->is<ListSymbol>()) {
    k = k->as<ListSymbol>()->item(0);
  }
  constexpr const size_t out_num = 2;
  if (!x->HasData()) {
    return GenList(SymbolPtrList(out_num, GenVList()));
  }
  DoNotEvalOnRun();
  auto result = x->symbols();
  if (!result.empty()) {
    result.back() = k;
  }
  return GenList(SymbolPtrList(out_num, GenList(std::move(result))));
}

REG_SYMBOL_OP_BUILDER("TopK").SetShapeDepend({DependOn::kShape, DependOn::kValue}).SetShapeFuncWith<TopK>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
