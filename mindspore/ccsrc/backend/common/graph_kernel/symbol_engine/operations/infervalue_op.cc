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
#include "backend/common/graph_kernel/symbol_engine/operations/infervalue_op.h"
#include <algorithm>
#include <utility>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "ops/normalize_dim_index.h"
#include "backend/common/graph_kernel/symbol_engine/operations/common_op.h"
#include "backend/common/graph_kernel/symbol_engine/operations/infershape_op.h"

namespace mindspore::graphkernel::symbol {
namespace ops::infervalue {
SymbolPtr RealValue::GenVarByShape(const IListSymbol &shape) {
  if (shape.HasData()) {
    // scalar value
    if (shape.size() == 0) {
      return GenVInt();
    }
    if (shape.AllHaveData() && shape.size() == 1) {
      return GenVIntList(LongToSize(shape.item(0)));
    }
  }
  return GenVList();
}

SymbolPtr RealValue::GenListVariables(const ListSymbol &list) {
  auto ilist = list.as<IListSymbol>();
  if (ilist != nullptr) {
    return GenVarByShape(*ilist);
  }
  if (!list.HasData()) {
    return ListSymbol::Make(shared_from_this());
  }
  SymbolPtrList result(list.size());
  (void)std::transform(list.symbols().begin(), list.symbols().end(), result.begin(), [this](const SymbolPtr &shape) {
    auto inner_list = shape->as<ListSymbol>();
    MS_EXCEPTION_IF_NULL(inner_list);
    return GenListVariables(*inner_list);
  });
  return ListSymbol::Make(std::move(result));
}

SymbolPtr RealValue::Eval() {
  auto v = input_as<InputSymbol>(0)->abstract()->GetValue();
  if (is_building() && v->isa<ValueAny>()) {
    OperationEmitter e;
    auto list = e.RealShape(input(0))->as_sptr<ListSymbol>();
    MS_EXCEPTION_IF_NULL(list);
    return GenListVariables(*list);
  }
  if (v->isa<ValueSequence>()) {
    return FromShape(GetValue<std::vector<int64_t>>(v), true);
  }
  if (v->isa<tensor::Tensor>()) {
    auto tensor_value = CheckAndConvertUtils::CheckTensorIntValue(v->ToString(), v, "RealValue");
    auto tensor = v->cast<tensor::TensorPtr>();
    return tensor->shape().empty() ? GenInt(tensor_value[0]) : FromShape(tensor_value, true);
  }
  if (v->isa<IntegerImm>()) {
    return GenInt(GetValue<int64_t>(v));
  }
  MS_LOG(EXCEPTION) << "Value should be one of {ValueSequence, Tensor, Integer}, but got " << v->ToString();
}

SymbolPtr NormalizeSlice::Eval() {
  auto data_shape = input_as<IListSymbol>(kIndex0);
  auto start = input(kIndex1);
  auto stop = input(kIndex2);
  auto step = input(kIndex3);
  auto tuple_index_axis = input_as<IntSymbol>(kIndex4)->value();
  auto tuple_index_types = ToShape(input_as<IListSymbol>(kIndex5));
  auto expand_dims_mask = input_as<IntSymbol>(kIndex6)->value();
  auto init_by_none = input_as<IListSymbol>(kIndex7);
  SymbolPtr dim_size;
  if (tuple_index_types.empty()) {
    dim_size = data_shape->symbol(0);
  } else {
    auto new_index_axis = mindspore::ops::NormalizeDimIndex::ConstNormalizeDimIndex(
      data_shape->size(), tuple_index_axis, tuple_index_types, expand_dims_mask);
    dim_size = data_shape->symbol(new_index_axis);
  }
  bool start_by_none_init = (init_by_none->item(kIndex0) == 1);
  bool stop_by_none_init = (init_by_none->item(kIndex1) == 1);
  bool step_by_none_init = (init_by_none->item(kIndex2) == 1);
  if (step_by_none_init) {
    step = GenInt(1);
  }
  auto step_int = step->as<IntSymbol>();
  MS_EXCEPTION_IF_NULL(step_int);
  bool unknown_output = false;
  if (start_by_none_init || stop_by_none_init) {
    if (step_int->is_positive()) {
      if (start_by_none_init) {
        start = GenInt(0);
      }
      if (stop_by_none_init) {
        stop = dim_size;
      }
    } else if (step_int->is_negative()) {
      if (start_by_none_init) {
        start = GenInt(-1);
      }
      if (stop_by_none_init) {
        // - (dim + 1)
        auto dim_plus_1 = Emit(std::make_shared<ScalarAdd>(dim_size, GenInt(1)));
        stop = Emit(std::make_shared<ScalarMul>(dim_plus_1, GenInt(-1)));
      }
    } else {
      // the sign of step is unknown
      unknown_output = true;
      if (start_by_none_init) {
        start = GenVInt();
      }
      if (stop_by_none_init) {
        stop = GenVInt();
      }
    }
  }
  if (!unknown_output) {
    DoNotEvalOnRun();
  }
  return ListSymbol::Make({GenList({start}), GenList({stop}), GenList({step})});
}

bool ShapeCalcBroadcastGradientArgs::NeedReduceAxis(const IntSymbolPtr xi, const IntSymbolPtr yi, bool *is_dyn) const {
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

SymbolPtr ShapeCalcBroadcastGradientArgs::Eval() {
  auto x = input_as<IListSymbol>(kIndex0);
  auto y = input_as<IListSymbol>(kIndex1);
  size_t shift = LongToSize(input_as<IntSymbol>(kIndex2)->value());
  if (!x->HasData() || !y->HasData()) {
    return ListSymbol::Make({GenVList(), GenVList()}, shared_from_this());
  }

  SymbolPtrList axis_x;
  SymbolPtrList axis_y;
  bool dyn_axis_x = false;
  bool dyn_axis_y = false;
  auto const1 = IntSymbol::Make(1);
  auto maxlen = std::max(x->size(), y->size());
  for (size_t i = maxlen; i > shift; i--) {
    auto x_i = (i > x->size() ? const1 : x->symbols()[x->size() - i]->as_sptr<IntSymbol>());
    auto y_i = (i > y->size() ? const1 : y->symbols()[y->size() - i]->as_sptr<IntSymbol>());
    MS_EXCEPTION_IF_NULL(x_i);
    MS_EXCEPTION_IF_NULL(y_i);
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
  return ListSymbol::Make({grad_axis_x, grad_axis_y}, shared_from_this());
}
}  // namespace ops::infervalue
}  // namespace mindspore::graphkernel::symbol
