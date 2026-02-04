/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/ring_attention_update.h"
#include <string>
#include <set>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "include/op_enum.h"

namespace mindspore {
namespace ops {
namespace {
void RingAttentionCheckRank(const ShapeVector &shape, const std::string &para_name, const std::string &layout_str,
                            size_t rank) {
  constexpr int64_t kValidLastDim = 8;
  if (MS_UNLIKELY(shape.size() != rank)) {
    MS_EXCEPTION(ValueError) << "For 'RingAttentionUpdate', the rank of '" << para_name << "' should be " << rank
                             << " while layout is '" << layout_str << "', but got: " << shape.size();
  }
  if (layout_str == "BNS8" || layout_str == "TN8") {
    if (shape.back() != abstract::TensorShape::kShapeDimAny && shape.back() != kValidLastDim) {
      MS_EXCEPTION(ValueError) << "For 'RingAttentionUpdate', the last dim of 'prev_softmax' must be 8, but got  "
                               << shape.back() << ".";
    }
  }
}

int64_t CheckDimSame(const int64_t dim1, const int64_t dim2, const std::string &dim_name) {
  if (dim1 != abstract::Shape::kShapeDimAny && dim2 != abstract::Shape::kShapeDimAny && dim1 != dim2) {
    MS_EXCEPTION(ValueError) << "For 'RingAttentionUpdate', the " << dim_name
                             << "-dim of inputs should be the consistent, but got " << dim1 << " and " << dim2
                             << "respectively.";
  }
  return dim2 == abstract::Shape::kShapeDimAny ? dim1 : dim2;
}

ShapeVector RingAttentionGetShapeInfo(const std::string &op_name, const ShapeVector &prev_attn_shape,
                                      const ShapeVector &prev_softmax_shape, const FASInputLayoutMode &layout) {
  constexpr size_t kRank3 = 3;
  constexpr size_t kRank4 = 4;
  int64_t B = abstract::TensorShape::kShapeDimAny;
  int64_t S = abstract::TensorShape::kShapeDimAny;
  int64_t H = abstract::TensorShape::kShapeDimAny;
  int64_t T = abstract::TensorShape::kShapeDimAny;
  int64_t N = abstract::TensorShape::kShapeDimAny;
  int64_t D = abstract::TensorShape::kShapeDimAny;

  switch (layout) {
    case FASInputLayoutMode::SBH: {
      RingAttentionCheckRank(prev_attn_shape, "prev_attn", "SBH", kRank3);
      RingAttentionCheckRank(prev_softmax_shape, "prev_softmax", "BNS8", kRank4);
      auto B_a = prev_attn_shape[kIndex1];
      auto B_s = prev_softmax_shape[kIndex0];
      B = CheckDimSame(B_a, B_s, "B");
      auto S_a = prev_attn_shape[kIndex0];
      auto S_s = prev_softmax_shape[kIndex2];
      S = CheckDimSame(S_a, S_s, "S");
      H = prev_attn_shape[kIndex2];
      N = prev_softmax_shape[kIndex1];
    } break;
    case FASInputLayoutMode::TND: {
      RingAttentionCheckRank(prev_attn_shape, "prev_attn", "TND", kRank3);
      RingAttentionCheckRank(prev_softmax_shape, "prev_softmax", "TN8", kRank3);
      auto T_a = prev_attn_shape[kIndex0];
      auto N_a = prev_attn_shape[kIndex1];
      D = prev_attn_shape[kIndex2];
      auto T_s = prev_softmax_shape[kIndex0];
      auto N_s = prev_softmax_shape[kIndex1];
      T = CheckDimSame(T_a, T_s, "T");
      N = CheckDimSame(N_a, N_s, "N");
    } break;
    default:
      MS_EXCEPTION(ValueError) << "For " << op_name << "the value of 'layout' must be SBH/TND.";
  }

  if (layout == FASInputLayoutMode::SBH &&
      (H != abstract::TensorShape::kShapeDimAny && N != abstract::TensorShape::kShapeDimAny && H % N != 0)) {
    MS_EXCEPTION(ValueError) << "For " << op_name
                             << ", 'H' must be divisible by 'N' while layout is 'SBH', but got H: " << H
                             << ", N: " << N;
  }
  return std::vector<int64_t>{S, B, H, T, N, D};
}
}  // namespace

ShapeArray RingAttentionUpdateFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  constexpr int64_t kSoftmaxLastDim = 8;
  ShapeVector prev_attn_shape = input_infos[kIndex0]->GetShape();
  ShapeVector prev_softmax_max_shape = input_infos[kIndex1]->GetShape();

  ShapeVector prev_attn_out_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                                     abstract::Shape::kShapeDimAny};
  ShapeVector softmax_sum_max_shape = {abstract::TensorShape::kShapeRankAny};

  auto layout_opt = input_infos[kIndex7]->GetScalarValue<int64_t>();
  if (MS_LIKELY(layout_opt.has_value())) {
    if (layout_opt.value() == FASInputLayoutMode::SBH) {
      softmax_sum_max_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                               abstract::Shape::kShapeDimAny, kSoftmaxLastDim};
    } else if (layout_opt.value() == FASInputLayoutMode::TND) {
      softmax_sum_max_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, kSoftmaxLastDim};
    } else {
      MS_EXCEPTION(ValueError) << "For " << op_name << ", the value of 'layout' must be SBH/TND.";
    }
  } else {
    return {prev_attn_out_shape, softmax_sum_max_shape, softmax_sum_max_shape};
  }
  if (MS_UNLIKELY(IsDynamicRank(prev_attn_shape) || IsDynamicRank(prev_softmax_max_shape))) {
    return {prev_attn_out_shape, softmax_sum_max_shape, softmax_sum_max_shape};
  }
  auto shape_info = RingAttentionGetShapeInfo(op_name, prev_attn_shape, prev_softmax_max_shape,
                                              static_cast<FASInputLayoutMode>(layout_opt.value()));
  if (layout_opt.value() == FASInputLayoutMode::SBH) {
    prev_attn_out_shape[kIndex0] = shape_info[kIndex0];
    prev_attn_out_shape[kIndex1] = shape_info[kIndex1];
    prev_attn_out_shape[kIndex2] = shape_info[kIndex2];
    softmax_sum_max_shape[kIndex0] = shape_info[kIndex1];
    softmax_sum_max_shape[kIndex1] = shape_info[kIndex4];
    softmax_sum_max_shape[kIndex2] = shape_info[kIndex0];
  } else {
    prev_attn_out_shape[kIndex0] = shape_info[kIndex3];
    prev_attn_out_shape[kIndex1] = shape_info[kIndex4];
    prev_attn_out_shape[kIndex2] = shape_info[kIndex5];
    softmax_sum_max_shape[kIndex0] = shape_info[kIndex3];
    softmax_sum_max_shape[kIndex1] = shape_info[kIndex4];
  }

  return {prev_attn_out_shape, softmax_sum_max_shape, softmax_sum_max_shape};
}

std::vector<TypeId> RingAttentionUpdateFuncImpl::InferType(const PrimitivePtr &primitive,
                                                           const InferInfoPtrList &input_infos) const {
  auto prev_attn_out_dtype = input_infos[kIndex0]->GetType();
  auto prev_softmax_max_dtype = input_infos[kIndex1]->GetType();
  if (!input_infos[kIndex6]->IsNone()) {
    auto actual_seq_qlend_type = input_infos[kIndex6]->GetType();
    if (actual_seq_qlend_type != kNumberTypeInt64) {
      MS_EXCEPTION(TypeError) << "For primitive[RingAttentionUpdate], 'actual_seq_qlen' dtype should be int64.";
    }
  }
  return {prev_attn_out_dtype, prev_softmax_max_dtype, prev_softmax_max_dtype};
}

std::set<int64_t> RingAttentionUpdateFuncImpl::GetValueDependArgIndices() const {
  // The aclnn kernelmod depends on the values of the elements at these three index positions,
  // which will be used for tensor-to-vector conversion.
  int64_t kActualaSeqQlenIndex = 6;
  return {kActualaSeqQlenIndex};
}
}  // namespace ops
}  // namespace mindspore
