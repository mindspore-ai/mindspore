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
#include "mindspore/core/ops/symbol_ops_impl/scalar_div.h"
#include "mindspore/core/ops/shape_calc.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API ShapeCalcBroadcastGradientArgs : public InferValueOp {
 public:
  ShapeCalcBroadcastGradientArgs(const SymbolPtr &inp1, const SymbolPtr &inp2, const SymbolPtr &shift)
      : InferValueOp({inp1, inp2, shift}) {}
  ~ShapeCalcBroadcastGradientArgs() override = default;
  MS_DECLARE_PARENT(ShapeCalcBroadcastGradientArgs, InferValueOp)

 protected:
  SymbolPtr Eval() override {
    auto x = input_as<ListSymbol>(kIndex0);
    auto y = input_as<ListSymbol>(kIndex1);
    size_t shift = LongToSize(input_as<IntSymbol>(kIndex2)->value());
    if (!x->HasData() || !y->HasData()) {
      return GenList({GenVList(), GenVList()});
    }

    SymbolPtrList axis_x;
    SymbolPtrList axis_y;
    bool dyn_axis_x = false;
    bool dyn_axis_y = false;
    auto const1 = IntSymbol::Make(1);
    auto maxlen = std::max(x->size(), y->size());
    for (size_t i = maxlen; i > shift; i--) {
      auto x_i = (i > x->size() ? const1 : x->item_as_sptr<IntSymbol>(x->size() - i));
      auto y_i = (i > y->size() ? const1 : y->item_as_sptr<IntSymbol>(y->size() - i));
      if (x_i->EqualsTo(y_i)) {
        continue;
      }
      if (!dyn_axis_x && NeedReduceAxis(x_i, y_i, &dyn_axis_x)) {
        axis_x.push_back(GenInt(SizeToLong(maxlen - i)));
      }
      if (!dyn_axis_y && NeedReduceAxis(y_i, x_i, &dyn_axis_y)) {
        axis_y.push_back(GenInt(SizeToLong(maxlen - i)));
      }
    }
    auto grad_axis_x = dyn_axis_x ? GenVList() : GenList(std::move(axis_x));
    auto grad_axis_y = dyn_axis_y ? GenVList() : GenList(std::move(axis_y));
    return GenList({grad_axis_x, grad_axis_y});
  }

  bool NeedReduceAxis(const IntSymbolPtr &xi, const IntSymbolPtr &yi, bool *is_dyn) const {
    // in BroadcastGradientArgs, the condition to reduce i of x is "x.shape[i] == 1",
    // when y.shape[i] == 1, reduce the x[i] is unnecessary but not wrong.
    if (xi->HasData()) {
      return xi->value() == 1;
    }
    if (xi->is_greater_than(1)) {
      return false;
    }
    if (yi->HasData() && yi->value() == 1) {
      return false;
    }
    *is_dyn = true;
    return false;
  }
};

class MS_CORE_API ShapeCalcReduceSumGrad : public InferValueOp {
 public:
  ShapeCalcReduceSumGrad(const SymbolPtr &inp, const SymbolPtr &axis) : InferValueOp({inp, axis}) {}
  ~ShapeCalcReduceSumGrad() override = default;
  MS_DECLARE_PARENT(ShapeCalcReduceSumGrad, InferValueOp)

 protected:
  SymbolPtr Eval() override {
    auto inp = input(kIndex0);
    if (!inp->HasData()) {
      return GenVList();
    }
    auto axis = input(kIndex1);
    auto keep_dims = BoolSymbol::Make(true);
    auto skip_mode = BoolSymbol::Make(false);
    auto r_shape = Emit(std::make_shared<Reduce>(inp, axis, keep_dims, skip_mode));
    MS_EXCEPTION_IF_NULL(r_shape);
    MS_EXCEPTION_IF_CHECK_FAIL(r_shape->HasData(), "r_shape should not be dynamic-rank");
    auto inp_list = inp->as<ListSymbol>();
    MS_EXCEPTION_IF_NULL(inp_list);
    auto r_shape_list = r_shape->as<ListSymbol>();
    MS_EXCEPTION_IF_NULL(r_shape_list);
    SymbolPtrList scaling(r_shape_list->size());
    for (size_t i = 0; i < r_shape_list->size(); i++) {
      scaling[i] = Emit(std::make_shared<ScalarDiv>(inp_list->item(i), r_shape_list->item(i)));
    }
    return GenList({r_shape, GenList(scaling)});
  }
};

// todo, add InferSymbolicValue interface into ShapeCalcFunctor
SymbolPtr ShapeCalcValueBuilder(OperationBuilder *b) {
  auto functor_attr = b->prim()->GetAttr(kAttrFunctor);
  MS_EXCEPTION_IF_NULL(functor_attr);
  auto functor = functor_attr->cast_ptr<ShapeCalcBaseFunctor>();
  MS_EXCEPTION_IF_NULL(functor);
  if (functor->name() == "ShapeCalc_reduce_shape_shapecalc") {
    auto input = b->GetInputShape(kIndex0);
    auto axis = b->GetInputValue(kIndex1);
    return b->Emit(std::make_shared<ShapeCalcReduceSumGrad>(input, axis));
  }
  if (functor->name() == "ShapeCalc_BroadcastGradientArgs") {
    auto inp1 = b->GetInputShape(kIndex0);
    auto inp2 = b->GetInputShape(kIndex1);
    auto shift = IntSymbol::Make(SizeToLong(GetValue<size_t>(functor->ToValue())));
    return b->Emit(std::make_shared<ShapeCalcBroadcastGradientArgs>(inp1, inp2, shift));
  }
  return nullptr;
}

REG_SYMBOL_OP_BUILDER("ShapeCalc")
  .SetValueDepend([](const PrimitivePtr &p) -> std::vector<DependOn> {
    auto value_depend_attr = p->GetAttr(mindspore::ops::kAttrValueDepend);
    MS_EXCEPTION_IF_NULL(value_depend_attr);
    auto value_depend = GetValue<std::vector<bool>>(value_depend_attr);
    std::vector<DependOn> depends(value_depend.size());
    (void)std::transform(value_depend.cbegin(), value_depend.cend(), depends.begin(),
                         [](bool v) { return v ? DependOn::kValue : DependOn::kShape; });
    return depends;
  })
  .SetValueFunc(ShapeCalcValueBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
