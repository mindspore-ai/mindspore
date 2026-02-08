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

#include "infer/ops_func_impl/speed_fusion_attention_grad.h"
#include <string>
#include "infer/ops_func_impl/speed_fusion_attention.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "include/op_enum.h"

namespace mindspore {
namespace ops {
namespace {
void SpeedFusionAttentionGradCheckShape(const InferInfoPtr &input, const std::string &op_name,
                                        const std::string &arg_name) {
  if (MS_UNLIKELY(input->IsDynamicRank())) {
    return;
  }
  auto shape = input->GetShape();
  constexpr size_t kDim3 = 3;
  constexpr size_t kDim4 = 4;
  MS_CHECK_VALUE(shape.size() == kDim3 || shape.size() == kDim4, "For " + op_name + ", the rank of '" + arg_name +
                                                                   "' must be 3 or 4, but got " +
                                                                   std::to_string(shape.size()) + ".");
}
}  // namespace

int32_t SpeedFusionAttentionGradFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                          const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &input_layout = input_infos[kIndex5];
  auto &padding_mask = input_infos[kIndex7];
  auto &keep_prob = input_infos[kIndex14];
  auto &sparse_mode = input_infos[kIndex24];
  auto &pse_type = input_infos[kIndex27];

  auto input_layout_opt = input_layout->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!input_layout_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  MS_CHECK_VALUE(
    input_layout_opt.value() == FASInputLayoutMode::BSH || input_layout_opt.value() == FASInputLayoutMode::SBH ||
      input_layout_opt.value() == FASInputLayoutMode::BNSD || input_layout_opt.value() == FASInputLayoutMode::BSND ||
      input_layout_opt.value() == FASInputLayoutMode::TND,
    "For " + op_name + ", the value of 'input_layout' must be BSH/SBH/BNSD/BSND/TND.");

  auto sparse_mode_opt = sparse_mode->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!sparse_mode_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  if (input_layout_opt.value() == FASInputLayoutMode::TND) {
    MS_CHECK_VALUE((sparse_mode_opt.value() >= SpeedFusionAttentionSparseMode::NO_MASK &&
                    sparse_mode_opt.value() < SpeedFusionAttentionSparseMode::PREFIX) ||
                     (sparse_mode_opt.value() > SpeedFusionAttentionSparseMode::PREFIX &&
                      sparse_mode_opt.value() <= SpeedFusionAttentionSparseMode::BAND_LEFT_UP_CAUSAL),
                   "For " + op_name + ", the value of 'sparse_mode' must be in range of [0, 5) or (5, 8], but got " +
                     std::to_string(sparse_mode_opt.value()) + ".");

    auto &actual_seq_qlen = input_infos[kIndex22];
    auto &actual_seq_kvlen = input_infos[kIndex23];
    MS_CHECK_VALUE(
      !actual_seq_qlen->IsNone() && !actual_seq_kvlen->IsNone(),
      "For " + op_name + ", actual_seq_qlen or actual_seq_kvlen can't be none while input_layout is 'TND'");
  } else {
    MS_CHECK_VALUE(sparse_mode_opt.value() >= SpeedFusionAttentionSparseMode::NO_MASK &&
                     sparse_mode_opt.value() <= SpeedFusionAttentionSparseMode::PREFIX_COMPRESS,
                   "For " + op_name + ", the value of 'sparse_mode' must be in range of [0, 6], but got " +
                     std::to_string(sparse_mode_opt.value()) + ".");
  }

  MS_CHECK_VALUE(padding_mask->IsNone(), "For " + op_name + ", padding_mask is not supported yet.");

  auto keep_prob_opt = keep_prob->GetScalarValue<float>();
  if (MS_UNLIKELY(!keep_prob_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  MS_CHECK_VALUE(keep_prob_opt.value() >= 0 && keep_prob_opt.value() <= 1,
                 "For " + op_name + ", the value of 'keep_prob' must be in range of [0, 1], but got " +
                   std::to_string(keep_prob_opt.value()) + ".");

  auto pse_type_opt = pse_type->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!pse_type_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  MS_CHECK_VALUE(pse_type_opt.value() >= 0 && pse_type_opt.value() <= 3,
                 "For " + op_name + ", the value of 'pse_type' must be in range of [0, 3], but got " +
                   std::to_string(pse_type_opt.value()) + ".");

  return OP_CHECK_SUCCESS;
}

ShapeArray SpeedFusionAttentionGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &query = input_infos[kIndex0];
  auto &key = input_infos[kIndex1];
  auto &value = input_infos[kIndex2];
  auto &dy = input_infos[kIndex3];
  auto dq_shape = query->GetShape();
  auto dk_shape = key->GetShape();
  auto dv_shape = value->GetShape();

  SpeedFusionAttentionGradCheckShape(query, op_name, "query");
  SpeedFusionAttentionGradCheckShape(key, op_name, "key");
  SpeedFusionAttentionGradCheckShape(value, op_name, "value");
  SpeedFusionAttentionGradCheckShape(dy, op_name, "dy");

  ShapeVector dpse_shape = {0};
  if (!input_infos[kIndex6]->IsNone()) {
    dpse_shape = input_infos[kIndex6]->GetShape();
  }

  return {dq_shape, dk_shape, dv_shape, dpse_shape};
}

std::vector<TypeId> SpeedFusionAttentionGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                const InferInfoPtrList &input_infos) const {
  auto query_dtype = input_infos[kIndex0]->GetType();
  auto key_dtype = input_infos[kIndex1]->GetType();
  auto value_dtype = input_infos[kIndex2]->GetType();
  TypeId dpse_dtype = query_dtype;
  if (!input_infos[kIndex6]->IsNone()) {
    dpse_dtype = input_infos[kIndex6]->GetType();
  }
  return {query_dtype, key_dtype, value_dtype, dpse_dtype};
}
}  // namespace ops
}  // namespace mindspore
