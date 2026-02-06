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
#include "mindspore/ops/infer/shape_calc.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include "mindspore/ops/infer/symbol_ops_impl/common.h"
#include "mindspore/ops/infer/symbol_ops_impl/reduce.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"
#include "abstract/symbolic_shape/symbol.h"
#include "utils/log_adapter.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API FunctorOperation : public InferValueOp {
 public:
  FunctorOperation(const ShapeCalcBaseFunctorPtr &functor, SymbolPtrList &&inputs, const SymbolPtr &out_hint)
      : InferValueOp(std::move(inputs)), functor_(functor), out_hint_(out_hint) {}
  ~FunctorOperation() override = default;
  MS_DECLARE_PARENT(FunctorOperation, InferValueOp)

 protected:
  SymbolPtr Eval() override { return out_hint_; }
  void EvalOnRun() override {
    auto [args, pos_idx] = GetCalcInputs();
    auto shape_array = functor_->Calc(args, pos_idx);
    SymbolPtrList res;
    res.reserve(shape_array.size());
    for (auto shape : shape_array) {
      SymbolPtrList shape_symbs;
      shape_symbs.reserve(shape.size());
      std::transform(shape.begin(), shape.end(), std::back_inserter(shape_symbs),
                     [this](int64_t d) { return GenInt(d); });
      res.push_back(GenList(std::move(shape_symbs)));
    }

    if (res.size() == 1) {
      output_->Update(res[0]);
      return;
    }

    output_->Update(GenList(std::move(res)));
  }

 private:
  std::pair<bool, std::vector<ListSymbolPtr>> IsSequenceSymbol(const SymbolPtr &sym) {
    if (!sym->is<ListSymbol>()) {
      return std::make_pair(false, std::vector<ListSymbolPtr>{});
    }
    auto list_sym = sym->as_sptr<ListSymbol>();
    MS_EXCEPTION_IF_CHECK_FAIL(list_sym->HasData(), "ListSymbol should have data in run status!");
    if (list_sym->size() == 0) {
      return std::make_pair(false, std::vector<ListSymbolPtr>{});
    }

    std::vector<ListSymbolPtr> elements;
    elements.reserve(list_sym->size());
    for (size_t i = 0; i < list_sym->size(); ++i) {
      if (!list_sym->item(i)->is<ListSymbol>()) {
        return std::make_pair(false, std::vector<ListSymbolPtr>{});
      }
      auto elem_sym = list_sym->item_as_sptr<ListSymbol>(i);
      elements.push_back(elem_sym);
    }
    return std::make_pair(true, std::move(elements));
  }

  ShapeVector ConvertListSymToShapeVec(const SymbolPtr &sym) {
    if (sym->is<IntSymbol>()) {
      return ShapeVector{sym->as_sptr<IntSymbol>()->value()};
    }

    auto list_sym = sym->as_sptr<ListSymbol>();
    MS_EXCEPTION_IF_CHECK_FAIL(list_sym->HasData(), "ListSymbol should have data in run status!");
    ShapeVector res;
    res.reserve(list_sym->size());
    for (size_t i = 0; i < list_sym->size(); ++i) {
      auto int_sym = list_sym->item_as_sptr<IntSymbol>(i);
      res.push_back(int_sym->value());
    }

    return res;
  }

  std::pair<ShapeArray, std::vector<std::vector<size_t>>> GetCalcInputs() {
    ShapeArray args;
    std::vector<std::vector<size_t>> pos_idx;
    auto num = input_num();
    for (size_t i = 0; i < num; ++i) {
      auto input_sym = input(i);
      std::vector<size_t> pos;
      size_t offset_base = args.size();
      if (auto [is_sequence, elements] = IsSequenceSymbol(input_sym); is_sequence) {
        pos.reserve(elements.size());
        for (const auto &elem : elements) {
          args.push_back(ConvertListSymToShapeVec(elem));
          pos.push_back(offset_base++);
        }
      } else {
        args.push_back(ConvertListSymToShapeVec(input_sym));
        pos.push_back(offset_base);
      }
      pos_idx.push_back(pos);
    }

    return std::make_pair(args, pos_idx);
  }

  ShapeCalcBaseFunctorPtr functor_;
  SymbolPtr out_hint_;
};

class OPS_API ShapeCalcBroadcastGradientArgs : public InferValueOp {
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

class OPS_API ShapeCalcReduceSumGrad : public InferValueOp {
 public:
  ShapeCalcReduceSumGrad(const SymbolPtr &inp, const SymbolPtr &axis, const SymbolPtr &skip_mode)
      : InferValueOp({inp, axis, skip_mode}) {}
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
    auto skip_mode = input(kIndex2);
    auto r_shape = Emit(std::make_shared<Reduce>(inp, axis, keep_dims, skip_mode));
    MS_EXCEPTION_IF_NULL(r_shape);
    MS_EXCEPTION_IF_CHECK_FAIL(r_shape->HasData(), "r_shape should not be dynamic-rank");
    auto inp_list = inp->as<ListSymbol>();
    auto r_shape_list = r_shape->as<ListSymbol>();
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
  auto functor = functor_attr->cast<ShapeCalcBaseFunctorPtr>();
  MS_EXCEPTION_IF_NULL(functor);
  if (functor->name() == "ShapeCalc_ReduceShape") {
    auto input = b->GetInputShape(kIndex0);
    auto axis = b->GetInputValue(kIndex1);
    auto skip_mode = BoolSymbol::Make(GetValue<bool>(functor->ToValue()));
    return b->Emit(std::make_shared<ShapeCalcReduceSumGrad>(input, axis, skip_mode));
  }
  if (functor->name() == "ShapeCalc_BroadcastGradientArgs") {
    auto inp1 = b->GetInputShape(kIndex0);
    auto inp2 = b->GetInputShape(kIndex1);
    auto shift = IntSymbol::Make(SizeToLong(GetValue<size_t>(functor->ToValue())));
    return b->Emit(std::make_shared<ShapeCalcBroadcastGradientArgs>(inp1, inp2, shift));
  }

  auto only_depend_shape_attr = b->prim()->GetAttr(kAttrOnlyDependShape);
  MS_EXCEPTION_IF_NULL(only_depend_shape_attr);
  auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
  auto num = b->input_num();
  SymbolPtrList inputs;
  inputs.reserve(num);
  for (size_t i = 0; i < num; ++i) {
    if (!only_depend_shape[i]) {
      inputs.push_back(b->GetInputValue(i));
    } else {
      inputs.push_back(b->GetInputShape(i));
    }
  }
  MS_EXCEPTION_IF_NULL(b->out_abstract());
  auto out_hint = BuildSymbolicValue(b->out_abstract());
  return b->Emit(std::make_shared<FunctorOperation>(functor, std::move(inputs), out_hint));
}

REG_SYMBOL_OP_BUILDER("ShapeCalc")
  .SetValueDepend([](const PrimitivePtr &p, size_t) -> std::vector<DependOn> {
    auto only_depend_shape_attr = p->GetAttr(kAttrOnlyDependShape);
    MS_EXCEPTION_IF_NULL(only_depend_shape_attr);
    auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
    std::vector<DependOn> depends(only_depend_shape.size());
    (void)std::transform(only_depend_shape.cbegin(), only_depend_shape.cend(), depends.begin(),
                         [](bool v) { return v ? DependOn::kShape : DependOn::kValue; });
    return depends;
  })
  .SetValueFunc(ShapeCalcValueBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
