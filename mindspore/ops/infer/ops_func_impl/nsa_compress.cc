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
#include "infer/ops_func_impl/nsa_compress.h"

namespace mindspore {
namespace ops {

ShapeArray NsaCompressFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &input_tensor = input_infos[kIndex0];
  auto input_shape = input_tensor->GetShape();

  if (input_tensor->IsDynamicRank()) {
    return {input_shape};
  }

  // Get compress_block_size and compress_stride (may be unknown at compile time)
  auto &compress_block_size_info = input_infos[kIndex2];
  auto &compress_stride_info = input_infos[kIndex3];
  auto &actual_seq_len_info = input_infos[kIndex4];

  auto compress_block_size_opt = compress_block_size_info->GetScalarValue<int64_t>();
  auto compress_stride_opt = compress_stride_info->GetScalarValue<int64_t>();

  // Handle actual_seq_len (tuple[list[int]]), allow unknown value at compile time
  std::vector<int64_t> actual_seq_len_vector;
  if (actual_seq_len_info->IsNone()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', 'actual_seq_len' must not be None.";
  }

  auto actual_seq_len_opt = actual_seq_len_info->GetArrayValue<int64_t>();
  bool has_known_seq = actual_seq_len_opt.has_value();
  if (has_known_seq) {
    auto actual_seq_len_arr = actual_seq_len_opt.value();
    if (!actual_seq_len_arr.HasUnknownValue()) {
      actual_seq_len_vector = actual_seq_len_arr.ToVector();
    } else {
      has_known_seq = false;
    }
  }

  // Decide whether we can compute compressed first dim
  bool can_calc_first_dim = has_known_seq && compress_block_size_opt.has_value() && compress_stride_opt.has_value();
  // If we cannot compute first dim precisely, still return [?, N, D]
  if (!can_calc_first_dim) {
    ShapeVector output_shape = {abstract::Shape::kShapeDimAny, input_shape[kIndex1], input_shape[kIndex2]};
    return {output_shape};
  }

  const int64_t compress_block_size = compress_block_size_opt.value();
  const int64_t compress_stride = compress_stride_opt.value();
  // Calculate compressed KV number following torch logic when both scalars and seq are known
  if (compress_stride <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', compress_stride must be greater than 0.";
  }

  int64_t compress_kv_num = 0;
  int64_t pre_seqlen = 0;
  for (size_t i = 0; i < actual_seq_len_vector.size(); i++) {
    int64_t cur_seq_len = actual_seq_len_vector[i] - pre_seqlen;
    if (cur_seq_len >= compress_block_size) {
      compress_kv_num += (cur_seq_len - compress_block_size + compress_stride) / compress_stride;
    }
    pre_seqlen += cur_seq_len;
  }

  ShapeVector output_shape = {compress_kv_num, input_shape[kIndex1], input_shape[kIndex2]};
  return {output_shape};
}

std::vector<TypeId> NsaCompressFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kIndex0]->GetType();
  return {input_type};
}

}  // namespace ops
}  // namespace mindspore
