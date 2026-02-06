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

#include "infer/ops_func_impl/speed_fusion_attention.h"
#include <string>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "include/op_enum.h"

namespace mindspore {
namespace ops {
namespace {
void SpeedFusionAttentionCheckRank(const std::string &op_name, const ShapeVector &query_shape,
                                   const std::string &input_layout_str, size_t rank) {
  if (MS_UNLIKELY(query_shape.size() != rank)) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ", the rank of query should be " << rank
                             << " while input_layout is '" << input_layout_str << "', but got " << query_shape.size();
  }
}

ShapeVector SpeedFusionAttentionGetShapeInfo(const std::string &op_name, const ShapeVector &query_shape,
                                             const FASInputLayoutMode &input_layout, const InferInfoPtr &q_head_num) {
  constexpr size_t kRank3 = 3;
  constexpr size_t kRank4 = 4;
  int64_t B = 0;
  int64_t Sq = 0;
  int64_t H = 0;
  int64_t T = 0;
  int64_t N = 0;
  int64_t D = 0;

  switch (input_layout) {
    case FASInputLayoutMode::BSH:
      SpeedFusionAttentionCheckRank(op_name, query_shape, "BSH", kRank3);
      B = query_shape[kIndex0];
      Sq = query_shape[kIndex1];
      H = query_shape[kIndex2];
      break;
    case FASInputLayoutMode::SBH:
      SpeedFusionAttentionCheckRank(op_name, query_shape, "SBH", kRank3);
      B = query_shape[kIndex1];
      Sq = query_shape[kIndex0];
      H = query_shape[kIndex2];
      break;
    case FASInputLayoutMode::BNSD:
      SpeedFusionAttentionCheckRank(op_name, query_shape, "BNSD", kRank4);
      B = query_shape[kIndex0];
      N = query_shape[kIndex1];
      Sq = query_shape[kIndex2];
      break;
    case FASInputLayoutMode::BSND:
      SpeedFusionAttentionCheckRank(op_name, query_shape, "BSND", kRank4);
      B = query_shape[kIndex0];
      N = query_shape[kIndex2];
      Sq = query_shape[kIndex1];
      break;
    case FASInputLayoutMode::TND:
      SpeedFusionAttentionCheckRank(op_name, query_shape, "TND", kRank3);
      T = query_shape[kIndex0];
      N = query_shape[kIndex1];
      D = query_shape[kIndex2];
      break;
    default:
      MS_EXCEPTION(ValueError) << "For " << op_name << "the value of 'input_layout' must be BSH/SBH/BNSD/BSND/TND.";
  }

  auto q_head_num_opt = q_head_num->GetScalarValue<int64_t>();
  if (q_head_num_opt.has_value()) {
    if ((input_layout == FASInputLayoutMode::BSH || input_layout == FASInputLayoutMode::SBH) &&
        (H != abstract::TensorShape::kShapeDimAny && H % q_head_num_opt.value() != 0)) {
      MS_EXCEPTION(ValueError) << "For " << op_name
                               << ", 'H' must be divisible by 'head_num' while input_layout is 'BSH' or 'SBH', but H:"
                               << H << ", head_num:" << q_head_num_opt.value();
    }

    if ((input_layout == FASInputLayoutMode::BNSD || input_layout == FASInputLayoutMode::BSND) &&
        (N != abstract::TensorShape::kShapeDimAny && N != q_head_num_opt.value())) {
      MS_EXCEPTION(ValueError) << "For " << op_name << ", 'N' must be equal to 'head_num', but got N:" << N
                               << ", head_num:" << q_head_num_opt.value();
    }
  }

  return std::vector<int64_t>{B, Sq, H, T, N, D};
}
}  // namespace

int32_t SpeedFusionAttentionFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &key = input_infos[kIndex1];
  auto &value = input_infos[kIndex2];
  auto &input_layout = input_infos[kIndex4];
  auto &padding_mask = input_infos[kIndex8];
  auto &keep_prob = input_infos[kIndex11];
  auto &sparse_mode = input_infos[kIndex18];
  auto &pse_type = input_infos[kIndex21];

  if (MS_UNLIKELY(key->IsDynamicRank() || value->IsDynamicRank())) {
    return OP_CHECK_RETRY;
  }

  auto key_shape = key->GetShape();
  auto value_shape = value->GetShape();
  MS_CHECK_VALUE(key_shape == value_shape,
                 "For " + op_name + ", the shape of 'value' should be equal to 'key',  but got key " +
                   ShapeVectorToString(key_shape) + ", value " + ShapeVectorToString(value_shape));

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

    auto &actual_seq_qlen = input_infos[kIndex16];
    auto &actual_seq_kvlen = input_infos[kIndex17];
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

ShapeArray SpeedFusionAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &query = input_infos[kIndex0];
  auto query_shape = query->GetShape();
  ShapeVector attention_out_shape = query_shape;
  ShapeVector softmax_sum_max_shape;
  ShapeVector softmax_out_shape = {0};
  ShapeVector gen_mask_out_shape = {};

  auto input_layout_opt = input_infos[kIndex4]->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!input_layout_opt.has_value())) {
    softmax_sum_max_shape = ShapeVector{abstract::TensorShape::kShapeRankAny};
    return {attention_out_shape, softmax_sum_max_shape, softmax_sum_max_shape, softmax_out_shape,
            gen_mask_out_shape,  gen_mask_out_shape,    gen_mask_out_shape};
  }

  constexpr int64_t kSoftmaxLastDim = 8;
  const auto idle_dim = abstract::TensorShape::kShapeDimAny;
  if (input_layout_opt.value() != FASInputLayoutMode::TND) {
    softmax_sum_max_shape = ShapeVector{idle_dim, idle_dim, idle_dim, kSoftmaxLastDim};
    auto head_num_opt = input_infos[kIndex3]->GetScalarValue<int64_t>();
    if (MS_LIKELY(head_num_opt.has_value())) {
      softmax_sum_max_shape[kIndex1] = head_num_opt.value();
    }
  } else {
    softmax_sum_max_shape = ShapeVector{idle_dim, idle_dim, kSoftmaxLastDim};
  }

  if (MS_LIKELY(!query->IsDynamicRank())) {
    auto shape_info = SpeedFusionAttentionGetShapeInfo(
      op_name, query_shape, static_cast<FASInputLayoutMode>(input_layout_opt.value()), input_infos[kIndex3]);
    constexpr int64_t kBIndex = 0;
    constexpr int64_t kSqIndex = 2;
    constexpr int64_t kTIndex = 0;
    constexpr int64_t kNIndex = 1;
    if (input_layout_opt.value() != FASInputLayoutMode::TND) {
      softmax_sum_max_shape[kBIndex] = shape_info[kIndex0];
      softmax_sum_max_shape[kSqIndex] = shape_info[kIndex1];
    } else {
      softmax_sum_max_shape[kTIndex] = shape_info[kIndex3];
      softmax_sum_max_shape[kNIndex] = shape_info[kIndex4];
    }
  }

  return {attention_out_shape, softmax_sum_max_shape, softmax_sum_max_shape, softmax_out_shape,
          gen_mask_out_shape,  gen_mask_out_shape,    gen_mask_out_shape};
}

std::vector<TypeId> SpeedFusionAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                                            const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto query_dtype = input_infos[kIndex0]->GetType();
  auto key_dtype = input_infos[kIndex1]->GetType();
  auto value_dtype = input_infos[kIndex2]->GetType();
  auto softmax_sum_max_dtype = kNumberTypeFloat32;
  auto gen_mask_out_dtype = kNumberTypeInt64;

  const std::set valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  auto inputs_dtype = std::vector<TypeId>{query_dtype, key_dtype, value_dtype};
  MS_CHECK_VALUE(
    std::all_of(inputs_dtype.begin(), inputs_dtype.end(),
                [&valid_types](const TypeId &dtype) { return valid_types.find(dtype) != valid_types.end(); }),
    "For " + op_name + ", the dtypes of 'query', 'key' and 'value' must be in (fp16, fp32, bf16).");

  if (!input_infos[kIndex9]->IsNone()) {
    auto atten_mask_type = input_infos[kIndex9]->GetType();
    if (atten_mask_type != kNumberTypeBool && atten_mask_type != kNumberTypeUInt8) {
      MS_EXCEPTION(ValueError) << "For " << op_name << ", the dtype of atten_mask should be bool or uint8, but got "
                               << TypeIdToString(atten_mask_type);
    }
  }

  return {query_dtype,        softmax_sum_max_dtype, softmax_sum_max_dtype, query_dtype,
          gen_mask_out_dtype, gen_mask_out_dtype,    gen_mask_out_dtype};
}
}  // namespace ops
}  // namespace mindspore
