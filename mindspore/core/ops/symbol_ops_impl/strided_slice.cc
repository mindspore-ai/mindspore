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
#include "mindspore/core/ops/symbol_ops_impl/scalar_max.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_min.h"

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
  SymbolPtr GetSlicingLengthForNegativeStrides(IntSymbolPtr start, IntSymbolPtr end, IntSymbolPtr strides,
                                               IntSymbolPtr x_dim);
  int CalcOutRank(int x_len, int slice_len) const {
    int out_rank = x_len;
    for (int j = 0; j < slice_len; j++) {
      if (new_axis_mask(j)) {
        out_rank++;
      } else if (shrink_axis_mask(j)) {
        out_rank--;
      }
    }
    return out_rank;
  }

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
  }
  MS_EXCEPTION_IF_CHECK_FAIL(
    begin->size() == end->size() && begin->size() == strides->size(),
    "For 'StridedSlice', the size of 'begin','end' and 'strides' should be equal. " + ToString());
  // refer to ComputeInferShape in "core/ops/strided_slice.cc"
  return ComputeInferShape(data_shape, begin, end, strides);
}

SymbolPtr StridedSlice::GetSlicingLengthForPositiveStrides(IntSymbolPtr start, IntSymbolPtr end, IntSymbolPtr strides,
                                                           IntSymbolPtr x_dim) {
  // the init start should be in range (-inf, x_dim)
  if (start == nullptr) {
    start = kSym0;
  } else if (start->is_negative()) {
    start = Emit(std::make_shared<ScalarAdd>(start, x_dim))->as_sptr<IntSymbol>();
  } else if (!start->is_greater_equal(0)) {
    // the "start" may be positive or negative, so the result length is unknown.
    return GenVInt();
  }
  start = Emit(std::make_shared<ScalarMax>(start, kSym0))->as_sptr<IntSymbol>();

  // the init "end" should be in range [-x_dim, inf)
  if (end == nullptr) {
    end = x_dim;
  } else if (end->is_negative()) {
    end = Emit(std::make_shared<ScalarAdd>(end, x_dim))->as_sptr<IntSymbol>();
  } else if (!end->is_greater_equal(0)) {
    // the "end" may be positive or negative, so the result length is unknown.
    return GenVInt();
  }
  end = Emit(std::make_shared<ScalarMin>(end, x_dim))->as_sptr<IntSymbol>();
  if ((*start) >= (*end)) {
    return kSym0;
  }
  if ((*start) <= (*end)) {
    // slice length = (end - start) / strides.  (to ceil)
    auto len = Emit(std::make_shared<ScalarSub>(end, start));
    return Emit(std::make_shared<ScalarCeilDiv>(len, strides));
  }
  return GenVInt();
}

SymbolPtr StridedSlice::GetSlicingLengthForNegativeStrides(IntSymbolPtr start, IntSymbolPtr end, IntSymbolPtr strides,
                                                           IntSymbolPtr x_dim) {
  // the init "start" should be in range [-x_dim, inf)
  if (start == nullptr) {
    start = kSymNeg1;
  } else if (start->is_greater_equal(0)) {
    // convert the "start" to [-n, -1]
    start = Emit(std::make_shared<ScalarSub>(start, x_dim))->as_sptr<IntSymbol>();
  } else if (!start->is_negative()) {
    // the "start" may be positive or negative, so the result length is unknown.
    return GenVInt();
  }
  start = Emit(std::make_shared<ScalarMin>(start, kSymNeg1))->as_sptr<IntSymbol>();

  // min_end = -x_dim - 1
  auto min_end = Emit(std::make_shared<ScalarSub>(kSymNeg1, x_dim))->as_sptr<IntSymbol>();
  // the init "end" should be in range (-inf, x_dim]
  if (end == nullptr) {
    end = min_end;
  } else if (end->is_greater_equal(0)) {
    end = Emit(std::make_shared<ScalarSub>(end, x_dim))->as_sptr<IntSymbol>();
  } else if (!end->is_negative()) {
    return GenVInt();
  }
  end = Emit(std::make_shared<ScalarMax>(end, min_end))->as_sptr<IntSymbol>();
  if ((*start) <= (*end)) {
    return kSym0;
  }
  if ((*start) >= (*end)) {
    // slice length = (end - start) / strides.  (to ceil)
    auto len = Emit(std::make_shared<ScalarSub>(end, start));
    return Emit(std::make_shared<ScalarCeilDiv>(len, strides));
  }
  return GenVInt();
}

SymbolPtr StridedSlice::ComputeInferShape(const ListSymbol *x_shape, const ListSymbol *begin_v, const ListSymbol *end_v,
                                          const ListSymbol *strides_v) {
  int slice_len = SizeToInt(begin_v->size());
  SymbolPtrList res_shape;
  int out_rank = CalcOutRank(SizeToInt(x_shape->size()), slice_len);
  res_shape.reserve(begin_v->size());
  int i = 0;  // used to visit x_shape
  int j = 0;  // used to visit slice info
  for (int k = 0; k < out_rank; k++, i++, j++) {
    auto x_dim_size = x_shape->item_as_sptr<IntSymbol>(static_cast<size_t>(i));
    MS_EXCEPTION_IF_NULL(x_dim_size);
    if (j >= slice_len) {
      (void)res_shape.emplace_back(x_dim_size);
      continue;
    }
    if (new_axis_mask(j)) {
      (void)res_shape.emplace_back(kSym1);
      i--;
      continue;
    }
    if (shrink_axis_mask(j)) {
      k--;
      continue;
    }
    IntSymbolPtr start = !begin_mask(j) ? begin_v->item_as_sptr<IntSymbol>(static_cast<size_t>(j)) : nullptr;
    IntSymbolPtr finish = !end_mask(j) ? end_v->item_as_sptr<IntSymbol>(static_cast<size_t>(j)) : nullptr;
    auto strides = strides_v->item_as_sptr<IntSymbol>(static_cast<size_t>(j));
    SymbolPtr slicing_len;
    if (strides->is_positive()) {
      slicing_len = GetSlicingLengthForPositiveStrides(start, finish, strides, x_dim_size);
    } else if (strides->is_negative()) {
      slicing_len = GetSlicingLengthForNegativeStrides(start, finish, strides, x_dim_size);
    } else {
      // unknown +/- of stride.
      slicing_len = GenVInt();
    }
    if (slicing_len == nullptr) {
      return nullptr;
    }
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
  auto begin_mask = b->GetInputOrAttr(kIndex4, kAttrBeginMask);
  MS_EXCEPTION_IF_NULL(begin_mask);
  auto end_mask = b->GetInputOrAttr(kIndex5, kAttrEndMask);
  MS_EXCEPTION_IF_NULL(end_mask);
  auto ellipsis_mask = b->GetInputOrAttr(kIndex6, kAttrEllipsisMask);
  MS_EXCEPTION_IF_NULL(ellipsis_mask);
  auto new_axis_mask = b->GetInputOrAttr(kIndex7, kAttrNewAxisMask);
  MS_EXCEPTION_IF_NULL(new_axis_mask);
  auto shrink_axis_mask = b->GetInputOrAttr(kIndex8, kAttrShrinkAxisMask);
  MS_EXCEPTION_IF_NULL(shrink_axis_mask);

  return b->Emit(std::make_shared<StridedSlice>(SymbolPtrList{data_shape, begin, end, strides, begin_mask, end_mask,
                                                              ellipsis_mask, new_axis_mask, shrink_axis_mask}));
}

// BuildValue of StridedSlice only support getting a single item.
SymbolPtr StridedSliceValueBuilder(OperationBuilder *b) {
  auto data = b->GetInputValue(kIndex0)->as_sptr_noexcept<ListSymbol>();
  if (data == nullptr || !data->HasData()) {
    return nullptr;
  }
  auto begin = b->GetInputValue(kIndex1);
  auto begin_list = begin->as_noexcept<ListSymbol>();
  int64_t idx = 0;
  if (begin_list != nullptr) {
    if (begin_list->size() != 1 || !begin_list->AllHaveData()) {
      return nullptr;
    }
    idx = AsInt(begin_list->item(0));
  } else {
    auto begin_int = begin->as_noexcept<IntSymbol>();
    if (begin_int == nullptr || !begin_int->HasData()) {
      return nullptr;
    }
    idx = begin_int->value();
  }
  auto shrink = b->GetInputOrAttr(kIndex8, kAttrShrinkAxisMask);
  if (shrink == nullptr || shrink->as<IntSymbol>()->value() != 1) {
    return nullptr;
  }
  return data->item(LongToSize(NormAxis(idx, data->size())));
}

REG_SYMBOL_OP_BUILDER("StridedSlice")
  .SetShapeDepend({DependOn::kShape, DependOn::kValue, DependOn::kValue, DependOn::kValue, DependOn::kValue,
                   DependOn::kValue, DependOn::kValue, DependOn::kValue, DependOn::kValue})
  .SetShapeFunc(StridedSliceShapeBuilder)
  .SetValueDependN<DependOn::kValue, 9>()
  .SetValueFunc(StridedSliceValueBuilder);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
