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
#include <set>
#include <memory>
#include <vector>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/nsa_compress_attention.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kSoftmaxMaxSize = 8;
}  // anonymous namespace

ShapeArray NsaCompressAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &query_tensor = input_infos[kIndex0];
  auto &key_tensor = input_infos[kIndex1];
  auto &value_tensor = input_infos[kIndex2];

  auto query_shape = query_tensor->GetShape();
  auto key_shape = key_tensor->GetShape();
  auto value_shape = value_tensor->GetShape();

  if (query_tensor->IsDynamicRank() || key_tensor->IsDynamicRank() || value_tensor->IsDynamicRank()) {
    return {query_shape, key_shape, query_shape, query_shape};
  }

  auto &head_num_info = input_infos[kIndex4];
  auto &select_block_count_info = input_infos[kIndex8];

  auto head_num_opt = head_num_info->GetScalarValue<int64_t>();
  auto select_block_count_opt = select_block_count_info->GetScalarValue<int64_t>();

  bool can_calc_shapes = head_num_opt.has_value() && select_block_count_opt.has_value();
  if (!can_calc_shapes) {
    const int64_t kShapeDimAny = abstract::Shape::kShapeDimAny;

    ShapeVector attention_out_shape = {query_shape[kIndex0], query_shape[kIndex1], value_shape[kIndex2]};
    ShapeVector topk_indices_shape = {query_shape[kIndex0], key_shape[kIndex1],
                                      select_block_count_opt.value_or(kShapeDimAny)};
    ShapeVector softmax_max_shape = {query_shape[kIndex0], query_shape[kIndex1], kSoftmaxMaxSize};
    ShapeVector softmax_sum_shape = {query_shape[kIndex0], query_shape[kIndex1], kSoftmaxMaxSize};
    return {attention_out_shape, topk_indices_shape, softmax_max_shape, softmax_sum_shape};
  }

  const int64_t head_num = head_num_opt.value();
  const int64_t select_block_count = select_block_count_opt.value();

  if (head_num <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', head_num must be greater than 0, but got: " << head_num;
  }
  if (select_block_count <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', select_block_count must be greater than 0, but got: " << select_block_count;
  }
  ShapeVector attention_out_shape = {query_shape[kIndex0], query_shape[kIndex1], value_shape[kIndex2]};
  ShapeVector topk_indices_shape = {query_shape[kIndex0], key_shape[kIndex1], select_block_count};
  ShapeVector softmax_max_shape = {query_shape[kIndex0], query_shape[kIndex1], kSoftmaxMaxSize};
  ShapeVector softmax_sum_shape = {query_shape[kIndex0], query_shape[kIndex1], kSoftmaxMaxSize};

  return {attention_out_shape, topk_indices_shape, softmax_max_shape, softmax_sum_shape};
}

std::vector<TypeId> NsaCompressAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                                            const InferInfoPtrList &input_infos) const {
  auto query_type = input_infos[kIndex0]->GetType();
  return {query_type, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32};
}

}  // namespace ops
}  // namespace mindspore
