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

#include "infer/ops_func_impl/sparse_flash_attention.h"

#include <string>
#include <memory>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/helper.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "op_def/op_enum.h"

namespace mindspore {
namespace ops {

namespace {
// Input indices
constexpr size_t kSFAInputQueryIndex = 0;
constexpr size_t kSFAInputKeyIndex = 1;
constexpr size_t kSFAInputValueIndex = 2;
constexpr size_t kSFAInputSparseIndicesIndex = 3;
constexpr size_t kSFAInputScaleValueIndex = 4;
constexpr size_t kSFAInputBlockTableIndex = 5;
constexpr size_t kSFAInputActualSeqLengthsQueryIndex = 6;
constexpr size_t kSFAInputActualSeqLengthsKvIndex = 7;
constexpr size_t kSFAInputQueryRopeIndex = 8;
constexpr size_t kSFAInputKeyRopeIndex = 9;
constexpr size_t kSFAInputSparseBlockSizeIndex = 10;
constexpr size_t kSFAInputLayoutQueryIndex = 11;
constexpr size_t kSFAInputLayoutKvIndex = 12;

// Output indices
constexpr size_t kSFAOutputAttentionOutIndex = 0;
constexpr size_t kSFAOutputSoftmaxMaxIndex = 1;
constexpr size_t kSFAOutputSoftmaxSumIndex = 2;
constexpr size_t kSFAOutputsNum = 3;

// Dimension constants
constexpr size_t kSFADim3 = 3;
constexpr size_t kSFADim4 = 4;

bool IsOptionalInputNone(const InferInfoPtr &input) { return input->IsNone(); }

int64_t GetLayoutInt(const InferInfoPtr &layout_info) {
  if (IsOptionalInputNone(layout_info)) {
    return FASInputLayoutMode::BSND;
  }
  auto layout_opt = layout_info->GetScalarValue<int64_t>();
  if (!layout_opt.has_value()) {
    return FASInputLayoutMode::BSND;
  }
  return layout_opt.value();
}
}  // namespace

ShapeArray SparseFlashAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();

  auto &query_info = input_infos[kSFAInputQueryIndex];
  auto &key_info = input_infos[kSFAInputKeyIndex];
  auto query_shape = query_info->GetShape();
  auto key_shape = key_info->GetShape();

  // Handle dynamic rank
  if (query_info->IsDynamicRank()) {
    ShapeVector dyn_rank{abstract::Shape::kShapeRankAny};
    return {dyn_rank, dyn_rank, dyn_rank};
  }

  int64_t layout_query = GetLayoutInt(input_infos[kSFAInputLayoutQueryIndex]);

  // Validate and infer output shape based on layout
  ShapeVector attention_out_shape;
  ShapeVector softmax_max_shape;
  ShapeVector softmax_sum_shape;

  if (layout_query == FASInputLayoutMode::TND) {
    // TND layout: query shape is [T1, N1, D]
    if (query_shape.size() != kSFADim3) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', when layout_query is TND, the rank of query must be 3, but got "
                               << query_shape.size();
    }
    attention_out_shape = {query_shape[0], query_shape[1], query_shape[2]};

    // softmax_max/softmax_sum shape: [N2, T1, N1/N2]
    if (key_shape.size() >= kSFADim3 && !IsDynamicRank(key_shape)) {
      int64_t n2 = key_shape[1];
      int64_t t1 = query_shape[0];
      int64_t n1 = query_shape[1];
      int64_t n1_div_n2 = (n1 > 0 && n2 > 0) ? (n1 / n2) : abstract::Shape::kShapeDimAny;
      softmax_max_shape = {n2, t1, n1_div_n2};
      softmax_sum_shape = {n2, t1, n1_div_n2};
    } else {
      softmax_max_shape = {abstract::Shape::kShapeDimAny, query_shape[0], abstract::Shape::kShapeDimAny};
      softmax_sum_shape = {abstract::Shape::kShapeDimAny, query_shape[0], abstract::Shape::kShapeDimAny};
    }
  } else {
    // BSND layout: query shape is [B, S1, N1, D]
    if (query_shape.size() != kSFADim4) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', when layout_query is BSND, the rank of query must be 4, but got "
                               << query_shape.size();
    }
    attention_out_shape = {query_shape[0], query_shape[1], query_shape[2], query_shape[3]};

    // softmax_max/softmax_sum shape: [B, N2, S1, N1/N2]
    if (key_shape.size() >= kSFADim4 && !IsDynamicRank(key_shape)) {
      int64_t b = query_shape[0];
      int64_t s1 = query_shape[1];
      int64_t n1 = query_shape[2];
      int64_t n2 = key_shape[2];
      int64_t n1_div_n2 = (n1 > 0 && n2 > 0) ? (n1 / n2) : abstract::Shape::kShapeDimAny;
      softmax_max_shape = {b, n2, s1, n1_div_n2};
      softmax_sum_shape = {b, n2, s1, n1_div_n2};
    } else {
      softmax_max_shape = {query_shape[0], abstract::Shape::kShapeDimAny, query_shape[1],
                           abstract::Shape::kShapeDimAny};
      softmax_sum_shape = {query_shape[0], abstract::Shape::kShapeDimAny, query_shape[1],
                           abstract::Shape::kShapeDimAny};
    }
  }

  return {attention_out_shape, softmax_max_shape, softmax_sum_shape};
}

std::vector<TypeId> SparseFlashAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                                            const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto query_type = input_infos[kSFAInputQueryIndex]->GetType();
  auto key_type = input_infos[kSFAInputKeyIndex]->GetType();
  auto value_type = input_infos[kSFAInputValueIndex]->GetType();

  // Validate that query, key, value have the same dtype
  if (query_type != key_type || query_type != value_type) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the dtypes of 'query', 'key' and 'value' must be the same, but got query: "
                            << TypeIdToString(query_type) << ", key: " << TypeIdToString(key_type)
                            << ", value: " << TypeIdToString(value_type);
  }

  std::vector<TypeId> output_types(kSFAOutputsNum);
  output_types[kSFAOutputAttentionOutIndex] = query_type;
  output_types[kSFAOutputSoftmaxMaxIndex] = kNumberTypeFloat32;
  output_types[kSFAOutputSoftmaxSumIndex] = kNumberTypeFloat32;

  return output_types;
}

}  // namespace ops
}  // namespace mindspore
