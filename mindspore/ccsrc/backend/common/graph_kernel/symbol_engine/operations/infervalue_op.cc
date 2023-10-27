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
  auto v = input_as<InputSymbol>(0)->abstract()->BuildValue();
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
}  // namespace ops::infervalue
}  // namespace mindspore::graphkernel::symbol
