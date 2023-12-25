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
#include "mindspore/core/ops/symbol_ops_impl/scalar_sub.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API StridedSlice : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~StridedSlice() override = default;
  MS_DECLARE_PARENT(StridedSlice, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
  SymbolPtr ComputeInferShape(const ListSymbol *x_shape, const ListSymbol *begin_v, const ListSymbol *end_v,
                              const ListSymbol *strides_v);
  SymbolPtr GetSlicingLengthForPositiveStrides(IntSymbolPtr start, IntSymbolPtr end, IntSymbolPtr strides,
                                               IntSymbolPtr x_dim);

  bool begin_mask(int bit) const { return ((begin_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  bool end_mask(int bit) const { return ((end_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  bool ellipsis_mask(int bit) const { return ((ellipsis_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  bool new_axis_mask(int bit) const { return ((new_axis_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  bool shrink_axis_mask(int bit) const { return ((shrink_axis_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  size_t begin_mask_{0};
  size_t end_mask_{0};
  size_t ellipsis_mask_{0};
  size_t new_axis_mask_{0};
  size_t shrink_axis_mask_{0};
  const ListSymbol *out_hint_{nullptr};
};

SymbolPtr StridedSlice::Eval() {
  auto data_shape = input_as<ListSymbol>(kIndex0);
  auto begin = input_as<ListSymbol>(kIndex1);
  auto end = input_as<ListSymbol>(kIndex2);
  auto strides = input_as<ListSymbol>(kIndex3);
  if (is_building()) {
    begin_mask_ = static_cast<size_t>(input_as<IntSymbol>(kIndex4)->value());
    end_mask_ = static_cast<size_t>(input_as<IntSymbol>(kIndex5)->value());
    ellipsis_mask_ = static_cast<size_t>(input_as<IntSymbol>(kIndex6)->value());
    if (ellipsis_mask_ != 0) {
      MS_LOG(DEBUG) << "StridedSlice infershape operation does not support ellipsis_mask yet.";
      return nullptr;
    }
    new_axis_mask_ = static_cast<size_t>(input_as<IntSymbol>(kIndex7)->value());
    shrink_axis_mask_ = static_cast<size_t>(input_as<IntSymbol>(kIndex8)->value());
    out_hint_ = input_as<ListSymbol>(kIndex9);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(begin->size() == end->size() && begin->size() == strides->size(),
                             "For StridedSlice, the size of 'begin','end' and 'strides' should be equal.");
  // refer to ComputeInferShape in "core/ops/strided_slice.cc"
  return ComputeInferShape(data_shape, begin, end, strides);
}

SymbolPtr StridedSlice::GetSlicingLengthForPositiveStrides(IntSymbolPtr start, IntSymbolPtr end, IntSymbolPtr strides,
                                                           IntSymbolPtr x_dim) {
  if (start->is_negative()) {
    start = Emit(std::make_shared<ScalarAdd>(start, x_dim))->as_sptr<IntSymbol>();
  }
  if (!start->is_greater_than(-1)) {
    return GenVInt();
  }

  if (end->is_negative()) {
    end = Emit(std::make_shared<ScalarAdd>(end, x_dim))->as_sptr<IntSymbol>();
  }
  if (!end->is_greater_than(-1)) {
    return GenVInt();
  }

  if ((*start) >= (*end)) {
    return GenInt(0);
  }
  if ((*start) < (*end)) {
    // length = (end - 1 - start) / strides + 1.  (to floor)
    auto t1 = Emit(std::make_shared<ScalarSub>(Emit(std::make_shared<ScalarSub>(end, GenInt(1))), start));
    auto t2 = Emit(std::make_shared<ScalarDiv>(t1, strides));
    return Emit(std::make_shared<ScalarAdd>(t2, GenInt(1)));
  }
  return GenVInt();
}

SymbolPtr StridedSlice::ComputeInferShape(const ListSymbol *x_shape, const ListSymbol *begin_v, const ListSymbol *end_v,
                                          const ListSymbol *strides_v) {
  int slice_len = SizeToInt(begin_v->size());
  SymbolPtrList res_shape;
  MS_EXCEPTION_IF_NULL(out_hint_);
  res_shape.reserve(out_hint_->size());
  int i = 0;
  int j = 0;
  for (int k = 0; k < static_cast<int>(out_hint_->size()); k++, i++, j++) {
    auto x_dim_size = x_shape->item_as_sptr<IntSymbol>(static_cast<size_t>(i));
    MS_EXCEPTION_IF_NULL(x_dim_size);
    if (j >= slice_len) {
      (void)res_shape.emplace_back(x_dim_size);
      continue;
    }
    if (shrink_axis_mask(j)) {
      k--;
      continue;
    }
    if (out_hint_->symbols()[k]->HasData()) {
      (void)res_shape.emplace_back(out_hint_->symbols()[k]);
      if (new_axis_mask(j)) {
        i--;
      }
      continue;
    }
    auto start = begin_v->item_as_sptr<IntSymbol>(j);
    auto finish = end_v->item_as_sptr<IntSymbol>(j);
    auto strides = strides_v->item_as_sptr<IntSymbol>(j);
    if (!strides->is_positive()) {
      // do not support negative strides yet.
      (void)res_shape.emplace_back(GenVInt());
      continue;
    }
    if (begin_mask(j)) {
      start = IntSymbol::Make(static_cast<int64_t>(0));
    }
    if (end_mask(j)) {
      finish = x_dim_size;
    }
    auto slicing_len = GetSlicingLengthForPositiveStrides(start, finish, strides, x_dim_size);
    (void)res_shape.emplace_back(std::move(slicing_len));
  }
  return ResultIntList(std::move(res_shape));
}

SymbolPtr StridedSliceShapeBuilder(OperationBuilder *b) {
  auto data_shape = b->GetInputShape(kIndex0);
  auto begin = b->GetInputValue(kIndex1);
  if (begin->is<IntSymbol>()) {
    begin = ListSymbol::Make({begin});
  }
  auto end = b->GetInputValue(kIndex2);
  if (end->is<IntSymbol>()) {
    end = ListSymbol::Make({end});
  }
  auto strides = b->GetInputValue(kIndex3);
  if (strides->is<IntSymbol>()) {
    strides = ListSymbol::Make({strides});
  }
  if (!data_shape->HasData() || !begin->HasData() || !end->HasData() || !strides->HasData()) {
    // not support dynamic rank
    return nullptr;
  }
  auto begin_mask = b->GetAttr(kAttrBeginMask);
  MS_EXCEPTION_IF_NULL(begin_mask);
  auto end_mask = b->GetAttr(kAttrEndMask);
  MS_EXCEPTION_IF_NULL(end_mask);
  auto ellipsis_mask = b->GetAttr(kAttrEllipsisMask);
  MS_EXCEPTION_IF_NULL(ellipsis_mask);
  auto new_axis_mask = b->GetAttr(kAttrNewAxisMask);
  MS_EXCEPTION_IF_NULL(new_axis_mask);
  auto shrink_axis_mask = b->GetAttr(kAttrShrinkAxisMask);
  MS_EXCEPTION_IF_NULL(shrink_axis_mask);

  auto out_hint = b->out_abstract()->GetShape()->BuildSymbolicShape();
  return b->Emit(std::make_shared<StridedSlice>(SymbolPtrList{
    data_shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out_hint}));
}

// BuildValue of StridedSlice only support getting a single item.
SymbolPtr StridedSliceValueBuilder(OperationBuilder *b) {
  auto data = b->GetInputValue(kIndex0)->as_sptr<ListSymbol>();
  if (data == nullptr || !data->HasData()) {
    return nullptr;
  }
  auto begin = b->GetInputValue(kIndex1);
  auto begin_list = begin->as<ListSymbol>();
  int64_t idx = 0;
  if (begin_list != nullptr) {
    if (begin_list->size() != 1 || !begin_list->AllHaveData()) {
      return nullptr;
    }
    idx = AsInt(begin_list->item(0));
  } else {
    auto begin_int = begin->as<IntSymbol>();
    if (begin_int == nullptr || !begin_int->HasData()) {
      return nullptr;
    }
    idx = begin_int->value();
  }
  auto shrink = b->GetAttr(kAttrShrinkAxisMask);
  if (shrink == nullptr || shrink->as<IntSymbol>()->value() != 1) {
    return nullptr;
  }
  return data->item(LongToSize(NormAxis(idx, data->size())));
}

REG_SYMBOL_OP_BUILDER("StridedSlice")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(StridedSliceShapeBuilder)
  .SetValueFunc(StridedSliceValueBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
