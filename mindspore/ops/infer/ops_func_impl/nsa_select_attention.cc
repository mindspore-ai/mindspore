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

#include "infer/ops_func_impl/nsa_select_attention.h"
#include <string>
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
namespace nsa_select_attention {
inline void CheckQkvValidation(const InferInfoPtr &input, const std::string &name) {
  constexpr size_t kTNDRank = 3;
  if (MS_UNLIKELY(!input->IsDynamicRank() && input->GetShape().size() != kTNDRank)) {
    MS_EXCEPTION(ValueError) << "For NsaSelectAttention, the layout of " << name
                             << " only support 'TND' now. So the rank of " << name << " must be " << kTNDRank
                             << ", but got " << input->GetShape().size() << ".";
  }
}
}  // namespace nsa_select_attention

int32_t NsaSelectAttentionFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto &actual_seq_qlen = input_infos[kIndex9];
  auto &actual_seq_kvlen = input_infos[kIndex10];
  if (MS_UNLIKELY(actual_seq_qlen->IsNone() || actual_seq_kvlen->IsNone())) {
    std::string error_msg = (actual_seq_qlen->IsNone() ? "'actual_seq_qlen'" : "");
    error_msg += (actual_seq_qlen->IsNone() && actual_seq_kvlen->IsNone() ? " and " : "");
    error_msg += (actual_seq_kvlen->IsNone() ? "'actual_seq_kvlen'" : "");
    MS_EXCEPTION(ValueError) << "For NsaSelectAttention, " << error_msg << " cannot be None, "
                             << "because the layout of 'query', 'key', and 'value' is now fixed to 'TND'.";
  }
  return OP_CHECK_SUCCESS;
}

ShapeArray NsaSelectAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  constexpr size_t kOutputRank = 3;
  constexpr size_t kSoftmaxMaxSumLastDim = 8;

  auto &query = input_infos[kIndex0];
  auto &key = input_infos[kIndex1];
  auto &value = input_infos[kIndex2];

  nsa_select_attention::CheckQkvValidation(query, "query");
  nsa_select_attention::CheckQkvValidation(key, "key");
  nsa_select_attention::CheckQkvValidation(value, "value");

  const auto query_shape = query->GetShape();
  const auto value_shape = value->GetShape();
  const auto is_query_dynamic_rank = query->IsDynamicRank();
  const auto is_value_dynamic_rank = value->IsDynamicRank();

  ShapeVector attention_out_shape;
  ShapeVector softmax_max_sum_shape;
  attention_out_shape.reserve(kOutputRank);
  softmax_max_sum_shape.reserve(kOutputRank);

  attention_out_shape.push_back(is_query_dynamic_rank ? abstract::TensorShape::kShapeDimAny : query_shape[kIndex0]);
  attention_out_shape.push_back(is_query_dynamic_rank ? abstract::TensorShape::kShapeDimAny : query_shape[kIndex1]);
  attention_out_shape.push_back(is_value_dynamic_rank ? abstract::TensorShape::kShapeDimAny : value_shape[kIndex2]);

  softmax_max_sum_shape.push_back(is_query_dynamic_rank ? abstract::TensorShape::kShapeDimAny : query_shape[kIndex0]);
  softmax_max_sum_shape.push_back(is_query_dynamic_rank ? abstract::TensorShape::kShapeDimAny : query_shape[kIndex1]);
  softmax_max_sum_shape.push_back(kSoftmaxMaxSumLastDim);

  return {attention_out_shape, softmax_max_sum_shape, softmax_max_sum_shape};
}

std::vector<TypeId> NsaSelectAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                                          const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType(), kNumberTypeFloat32, kNumberTypeFloat32};
}
}  // namespace ops
}  // namespace mindspore
