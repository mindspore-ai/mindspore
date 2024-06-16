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
#include "mindspore/core/ops/symbol_ops_impl/common.h"
#include "mindspore/core/ops/symbol_ops_impl/reduce.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API LayerNorm : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  LayerNorm(const SymbolPtr &x, const SymbolPtr &begin_axis) : InferShapeOp({x, begin_axis}) {}
  ~LayerNorm() override = default;
  MS_DECLARE_PARENT(LayerNorm, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr LayerNorm::Eval() {
  auto x = input_as<ListSymbol>(0);
  auto begin_axis_sym = input_as<IntSymbol>(1);
  if (!begin_axis_sym->HasData()) {
    auto mean_var = x->HasData() ? GenVIntList(x->size()) : GenVList();
    return GenList({input(0), mean_var, mean_var});
  }
  int64_t begin_axis = begin_axis_sym->value();
  if (!x->HasData() && begin_axis > 0) {
    auto mean_var = GenVList();
    return GenList({input(0), mean_var, mean_var});
  }
  DoNotEvalOnRun();
  SymbolPtr axis = nullptr;
  if (begin_axis < 0) {
    axis = input(1);
  } else if (begin_axis == 0) {
    axis = GenList({});
  } else {
    std::vector<int64_t> vec(LongToSize(static_cast<int64_t>(x->size()) - begin_axis));
    std::iota(vec.begin(), vec.end(), begin_axis);
    axis = IntValues2Symbol(vec, shared_from_this());
  }
  auto mean_var = Emit(std::make_shared<Reduce>(input(0), axis, BoolSymbol::Make(true), BoolSymbol::Make(false)));
  return GenList({input(0), mean_var, mean_var});
}

class LayerNormExt : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~LayerNormExt() override = default;
  MS_DECLARE_PARENT(LayerNormExt, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

SymbolPtr LayerNormExt::Eval() {
  auto x = input_as_sptr<ListSymbol>(kIndex0);
  auto norm_shape = input_as<ListSymbol>(kIndex1);
  if (!x->HasData() || !norm_shape->HasData()) {
    return GenVList();
  }
  DoNotEvalOnRun();
  return Emit(std::make_shared<LayerNorm>(x, GenInt(SizeToLong(x->size() - norm_shape->size()))));
}

REG_SYMBOL_OP_BUILDER("LayerNorm")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kNone, DependOn::kValue})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto x = b->GetInputShape(kIndex0);
    auto begin_axis = b->GetInputOrAttr(kIndex3, kAttrBeginNormAxis);
    MS_EXCEPTION_IF_NULL(begin_axis);
    return b->Emit(std::make_shared<LayerNorm>(x, begin_axis));
  });
REG_SYMBOL_OP_BUILDER("LayerNormV3")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kNone, DependOn::kValue})
  .SetShapeFuncWith<LayerNorm>();

REG_SYMBOL_OP_BUILDER("LayerNormExt")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue})
  .SetShapeFuncWith<LayerNormExt>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
