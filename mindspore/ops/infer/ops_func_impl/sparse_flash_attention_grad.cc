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

#include "infer/ops_func_impl/sparse_flash_attention_grad.h"

#include <string>
#include <memory>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/helper.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace ops {

namespace {
// Input indices for grad op
constexpr size_t kSFAGradInputQueryIndex = 0;
constexpr size_t kSFAGradInputKeyIndex = 1;
constexpr size_t kSFAGradInputValueIndex = 2;
constexpr size_t kSFAGradInputSparseIndicesIndex = 3;
constexpr size_t kSFAGradInputDOutIndex = 4;
constexpr size_t kSFAGradInputOutIndex = 5;
constexpr size_t kSFAGradInputSoftmaxMaxIndex = 6;
constexpr size_t kSFAGradInputSoftmaxSumIndex = 7;
constexpr size_t kSFAGradInputScaleValueIndex = 8;
constexpr size_t kSFAGradInputSparseBlockSizeIndex = 9;
constexpr size_t kSFAGradInputQueryRopeIndex = 10;
constexpr size_t kSFAGradInputKeyRopeIndex = 11;

// Output indices
constexpr size_t kSFAGradOutputDQueryIndex = 0;
constexpr size_t kSFAGradOutputDKeyIndex = 1;
constexpr size_t kSFAGradOutputDValueIndex = 2;
constexpr size_t kSFAGradOutputDQueryRopeIndex = 3;
constexpr size_t kSFAGradOutputDKeyRopeIndex = 4;
constexpr size_t kSFAGradOutputsNum = 5;

bool IsGradOptionalInputNone(const InferInfoPtr &input) { return input->IsNone(); }
}  // namespace

ShapeArray SparseFlashAttentionGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);

  auto &query_info = input_infos[kSFAGradInputQueryIndex];
  auto &key_info = input_infos[kSFAGradInputKeyIndex];
  auto &value_info = input_infos[kSFAGradInputValueIndex];
  auto &query_rope_info = input_infos[kSFAGradInputQueryRopeIndex];
  auto &key_rope_info = input_infos[kSFAGradInputKeyRopeIndex];

  auto query_shape = query_info->GetShape();
  auto key_shape = key_info->GetShape();
  auto value_shape = value_info->GetShape();

  // Handle dynamic rank
  if (query_info->IsDynamicRank()) {
    ShapeVector dyn_rank{abstract::Shape::kShapeRankAny};
    return {dyn_rank, dyn_rank, dyn_rank, dyn_rank, dyn_rank};
  }

  // d_query has same shape as query
  ShapeVector d_query_shape = query_shape;
  // d_key has same shape as key
  ShapeVector d_key_shape = key_shape;
  // d_value has same shape as value
  ShapeVector d_value_shape = value_shape;

  // d_query_rope has same shape as query_rope (or empty if None)
  ShapeVector d_query_rope_shape;
  if (!IsGradOptionalInputNone(query_rope_info)) {
    d_query_rope_shape = query_rope_info->GetShape();
  } else {
    d_query_rope_shape = {0};
  }

  // d_key_rope has same shape as key_rope (or empty if None)
  ShapeVector d_key_rope_shape;
  if (!IsGradOptionalInputNone(key_rope_info)) {
    d_key_rope_shape = key_rope_info->GetShape();
  } else {
    d_key_rope_shape = {0};
  }

  return {d_query_shape, d_key_shape, d_value_shape, d_query_rope_shape, d_key_rope_shape};
}

std::vector<TypeId> SparseFlashAttentionGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                const InferInfoPtrList &input_infos) const {
  auto query_type = input_infos[kSFAGradInputQueryIndex]->GetType();
  auto key_type = input_infos[kSFAGradInputKeyIndex]->GetType();
  auto value_type = input_infos[kSFAGradInputValueIndex]->GetType();

  std::vector<TypeId> output_types(kSFAGradOutputsNum);
  output_types[kSFAGradOutputDQueryIndex] = query_type;
  output_types[kSFAGradOutputDKeyIndex] = key_type;
  output_types[kSFAGradOutputDValueIndex] = value_type;

  // d_query_rope and d_key_rope have same dtype as query
  auto &query_rope_info = input_infos[kSFAGradInputQueryRopeIndex];
  auto &key_rope_info = input_infos[kSFAGradInputKeyRopeIndex];

  if (!IsGradOptionalInputNone(query_rope_info)) {
    output_types[kSFAGradOutputDQueryRopeIndex] = query_rope_info->GetType();
  } else {
    output_types[kSFAGradOutputDQueryRopeIndex] = query_type;
  }

  if (!IsGradOptionalInputNone(key_rope_info)) {
    output_types[kSFAGradOutputDKeyRopeIndex] = key_rope_info->GetType();
  } else {
    output_types[kSFAGradOutputDKeyRopeIndex] = key_type;
  }

  return output_types;
}

}  // namespace ops
}  // namespace mindspore
