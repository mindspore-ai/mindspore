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
SymbolPtr RealValue::GenVarByShape(const IListSymbol &shape, const TypePtr &type_ptr) {
  if (shape.HasData()) {
    // scalar value
    if (shape.size() == 0) {
      if (type_ptr->generic_type_id() == kNumberTypeBool) {
        return BoolSymbol::Make(shared_from_this());
      }
      if (type_ptr->generic_type_id() == kNumberTypeFloat) {
        return FloatSymbol::Make(shared_from_this());
      }
      return GenVInt();
    }
    if (shape.AllHaveData() && shape.size() == 1) {
      SymbolPtrList list(LongToSize(shape.item(0)));
      if (type_ptr->generic_type_id() == kNumberTypeBool) {
        std::generate(list.begin(), list.end(), [this]() { return BoolSymbol::Make(shared_from_this()); });
      } else if (type_ptr->generic_type_id() == kNumberTypeFloat) {
        std::generate(list.begin(), list.end(), [this]() { return FloatSymbol::Make(shared_from_this()); });
      } else {
        std::generate(list.begin(), list.end(), [this]() { return this->GenVInt(); });
      }
      return GenList(std::move(list));
    }
  }
  return GenVList();
}

SymbolPtr RealValue::GenListVariables(const ListSymbol &list, const TypePtr &type_ptr) {
  auto ilist = list.as<IListSymbol>();
  if (ilist != nullptr) {
    return GenVarByShape(*ilist, type_ptr);
  }
  if (!list.HasData()) {
    return ListSymbol::Make(shared_from_this());
  }
  SymbolPtrList result(list.size());
  auto tup = type_ptr->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(tup);
  auto inner_type = tup->elements()[0];
  bool all_int = true;
  (void)std::transform(list.symbols().begin(), list.symbols().end(), result.begin(),
                       [this, inner_type, &all_int](const SymbolPtr &shape) {
                         auto inner_list = shape->as<ListSymbol>();
                         MS_EXCEPTION_IF_NULL(inner_list);
                         auto ret = GenListVariables(*inner_list, inner_type);
                         all_int = all_int && ret->is<IntSymbol>();
                         return ret;
                       });
  return all_int ? IListSymbol::Make(std::move(result)) : ListSymbol::Make(std::move(result));
}

SymbolPtr RealValue::ParseValueSequence(const ValueSequeuePtr &seq) {
  SymbolPtrList result(seq->size());
  bool all_int = true;
  (void)std::transform(seq->value().begin(), seq->value().end(), result.begin(), [&all_int, this](const ValuePtr &v) {
    all_int = all_int && v->isa<IntegerImm>();
    return ParseConstValue(v);
  });
  return all_int ? IListSymbol::Make(std::move(result)) : ListSymbol::Make(std::move(result));
}

SymbolPtr RealValue::ParseConstValue(const ValuePtr &v) {
  if (v->isa<ValueSequence>()) {
    return ParseValueSequence(v->cast<ValueSequeuePtr>());
  }
  if (v->isa<tensor::Tensor>()) {
    auto tensor_value = CheckAndConvertUtils::CheckTensorIntValue(v->ToString(), v, "RealValue");
    auto tensor = v->cast<tensor::TensorPtr>();
    return tensor->shape().empty() ? GenInt(tensor_value[0]) : FromShape(tensor_value, true);
  }
  if (v->isa<IntegerImm>()) {
    return v->isa<Int64Imm>() ? GenInt(GetValue<int64_t>(v)) : GenInt(static_cast<int64_t>(GetValue<int32_t>(v)));
  }
  if (v->isa<BoolImm>()) {
    return BoolSymbol::Make(GetValue<bool>(v), shared_from_this());
  }
  if (v->isa<FloatImm>()) {
    return FloatSymbol::Make((v->isa<FP64Imm>() ? GetValue<double>(v) : static_cast<double>(GetValue<float>(v))),
                             shared_from_this());
  }
  if (v->isa<StringImm>()) {
    return StrSymbol::Make(GetValue<std::string>(v), shared_from_this());
  }
  MS_LOG(EXCEPTION)
    << "Value should be one of {ValueSequence, Tensor, IntegerImm, BoolImm, FloatImm, StringImm}, but got "
    << v->ToString();
}

SymbolPtr RealValue::Eval() {
  auto v = input_as<InputSymbol>(0)->abstract()->GetValue();
  if (is_building() && v->isa<ValueAny>()) {
    OperationEmitter e;
    auto list = e.RealShape(input(0))->as_sptr<ListSymbol>();
    MS_EXCEPTION_IF_NULL(list);
    auto type_ptr = input_as<InputSymbol>(0)->abstract()->BuildType();
    return GenListVariables(*list, type_ptr);
  }
  return ParseConstValue(v);
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
