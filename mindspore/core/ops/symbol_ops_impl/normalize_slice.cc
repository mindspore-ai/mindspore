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
#include "mindspore/core/ops/symbol_ops_impl/scalar_add.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_mul.h"
#include "mindspore/core/ops/normalize_dim_index.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API NormalizeSlice : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  ~NormalizeSlice() override = default;
  MS_DECLARE_PARENT(NormalizeSlice, InferValueOp)

 protected:
  SymbolPtr Eval() override {
    auto data_shape = input_as<ListSymbol>(kIndex0);
    auto start = input(kIndex1);
    auto stop = input(kIndex2);
    auto step = input(kIndex3);
    auto tuple_index_axis = input_as<IntSymbol>(kIndex4)->value();
    auto tuple_index_types = ToShape(input(kIndex5));
    auto expand_dims_mask = input_as<IntSymbol>(kIndex6)->value();
    auto init_by_none = input_as<ListSymbol>(kIndex7);
    SymbolPtr dim_size;
    if (tuple_index_types.empty()) {
      dim_size = data_shape->item(0);
    } else {
      auto new_index_axis = mindspore::ops::NormalizeDimIndex::ConstNormalizeDimIndex(
        data_shape->size(), tuple_index_axis, tuple_index_types, expand_dims_mask);
      dim_size = data_shape->item(new_index_axis);
    }
    bool start_by_none_init = (AsInt(init_by_none->item(kIndex0)) == 1);
    bool stop_by_none_init = (AsInt(init_by_none->item(kIndex1)) == 1);
    bool step_by_none_init = (AsInt(init_by_none->item(kIndex2)) == 1);
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
    return GenList({GenList({start}), GenList({stop}), GenList({step})});
  }
};

SymbolPtr NormalizeSliceValueBuilder(OperationBuilder *b) {
  auto data_shape = b->GetInputShape(kIndex0);
  if (!data_shape->HasData()) {
    // does not support dynamic rank
    return nullptr;
  }
  auto start = b->GetInputValue(kIndex1);
  auto stop = b->GetInputValue(kIndex2);
  auto step = b->GetInputValue(kIndex3);
  auto tuple_index_axis = b->GetAttr(kAttrTupleIndexAxis);
  auto tuple_index_types = b->GetAttr(kAttrTupleIndexTypes);
  auto expand_dims_mask = b->GetAttr(kAttrExpandDimsMask);
  auto init_by_none = b->GetAttr(kAttrInitByNone);
  return b->Emit(std::make_shared<NormalizeSlice>(
    SymbolPtrList{data_shape, start, stop, step, tuple_index_axis, tuple_index_types, expand_dims_mask, init_by_none}));
}

REG_SYMBOL_OP_BUILDER("NormalizeSlice")
  .SetValueDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue, DependOn::kValue})
  .SetValueFunc(NormalizeSliceValueBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
