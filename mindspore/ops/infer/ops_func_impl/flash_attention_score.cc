/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/flash_attention_score.h"

#include <string>
#include <map>
#include <memory>
#include <array>
#include <algorithm>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "primitive/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "mindapi/helper.h"
#include "ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
constexpr size_t kFlashAttentionScoreSoftmaxLastDim = 8;
constexpr size_t kInputFlashAttentionScoreQueryBSHRank = 3;
constexpr size_t kInputFlashAttentionScoreQuerySBHRank = 3;
constexpr size_t kInputFlashAttentionScoreQueryTNDRank = 3;
constexpr size_t kInputFlashAttentionScoreQueryBNSDRank = 4;
constexpr size_t kInputFlashAttentionScoreQueryBSNDRank = 4;
constexpr auto kEnableRingAttention = "enable_ring_attention";
constexpr auto kEnableFlashSP = "enable_flash_sp";
constexpr auto kEnableRASendRecv = "enable_ra_send_recv";

// None indicates that the optional input is not passed
bool IsFlashAttentionScoreOptionalInputNotPass(const InferInfoPtr &input) { return input->IsNone(); }

static inline void ValidateKeepProbAndDropMask(const InferInfoPtrList &input_infos, const std::string &op_name) {
  auto keep_prob_opt = input_infos[kFlashAttentionScoreInputKeepProbIndex]->GetScalarValue<float>();
  if (!keep_prob_opt.has_value()) {
    return;
  }
  const auto keep_prob = keep_prob_opt.value();
  if (keep_prob > 1 || keep_prob <= 0) {
    MS_LOG(EXCEPTION) << op_name << ": attribute `keep_prob` must be a floating point number in (0, 1], but got "
                      << keep_prob;
  }

  if (!IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputDropMaskIndex])) {
    const auto drop_mask_type = input_infos[kFlashAttentionScoreInputDropMaskIndex]->GetType();
    CheckAndConvertUtils::CheckTypeIdValid("drop_mask", drop_mask_type, {kNumberTypeUInt8}, op_name);
  }
}

void CheckFlashAttentionScoreSparseMode(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                        const std::vector<int64_t> &shape_info, int64_t q_head_num) {
  auto op_name = primitive->name();
  auto sparse_mode_opt = input_infos[kFlashAttentionScoreInputSparseModeIndex]->GetScalarValue<int64_t>();
  if (sparse_mode_opt.has_value()) {
    if (primitive->HasAttr(kEnableRingAttention)) {
      auto enable_ring_attention_valueptr = primitive->GetAttr(kEnableRingAttention);
      if (!(enable_ring_attention_valueptr->isa<BoolImm>())) {
        MS_LOG(EXCEPTION) << "For '" << op_name << "', the attribute 'enable_ring_attention' must be a bool";
      }
    }
    if (primitive->HasAttr(kEnableRASendRecv)) {
      auto enable_ra_sendrecv_valueptr = primitive->GetAttr(kEnableRASendRecv);
      if (!(enable_ra_sendrecv_valueptr->isa<BoolImm>())) {
        MS_LOG(EXCEPTION) << "For '" << op_name << "', the attribute 'enable_ra_send_recv' must be a bool";
      }
    }

    if (primitive->HasAttr(kEnableFlashSP)) {
      auto enable_flash_sp_valueptr = primitive->GetAttr(kEnableFlashSP);
      if (!(enable_flash_sp_valueptr->isa<BoolImm>())) {
        MS_LOG(EXCEPTION) << "For '" << op_name << "', the attribute 'enable_flash_sp' must be a bool";
      }
    }
  }
}

ShapeArray ConstructInferShape(const ShapeVector &softmax_shape, const ShapeVector &query_shape,
                               const ShapeVector &key_shape, const ShapeVector &value_shape,
                               std::optional<int64_t> input_layout = std::nullopt) {
  auto output_shape = query_shape;
  if (input_layout.has_value() && !IsDynamicRank(query_shape) && !IsDynamicRank(key_shape) &&
      !IsDynamicRank(value_shape)) {
    auto input_layout_pair = layoutMap.find(input_layout.value());
    if (input_layout_pair == layoutMap.end()) {
      MS_LOG(EXCEPTION) << "FlashAttentionScore: unsupported layout: " << input_layout.value();
    }
    const std::string &input_layout_str = input_layout_pair->second;
    if (input_layout_str.find("D") != std::string::npos) {
      auto head_dim_index = input_layout_str.find("D");
      auto value_head_dim = value_shape.at(head_dim_index);
      output_shape.at(head_dim_index) = value_head_dim;
    } else {
      auto hidden_dim_index = input_layout_str.find("H");
      if (hidden_dim_index == std::string::npos) {
        MS_LOG(EXCEPTION) << "FlashAttentionScore: cannot find the head_dim or hidden dimension from layout "
                          << input_layout_str;
      }
      auto query_hidden_size = query_shape.at(hidden_dim_index);
      auto key_hidden_size = key_shape.at(hidden_dim_index);
      auto value_hidden_size = value_shape.at(hidden_dim_index);
      auto output_hidden_size = query_hidden_size / key_hidden_size * value_hidden_size;
      output_shape.at(hidden_dim_index) = output_hidden_size < 0 ? abstract::Shape::kShapeDimAny : output_hidden_size;
    }
  }
  return ShapeArray{softmax_shape, softmax_shape, ShapeVector{1}, output_shape};
}

std::vector<int64_t> GetFASInfoFromInputLayout(int64_t input_layout, int64_t q_head_num, const std::string &op_name,
                                               const ShapeVector &query_shape, const ShapeVector &key_shape,
                                               const ShapeVector &value_shape) {
  int64_t batch_size = -1;
  int64_t q_seq_len = -1;
  int64_t kv_seq_len = -1;
  int64_t kv_head_num = -1;
  if (query_shape.size() != key_shape.size() || query_shape.size() != value_shape.size()) {
    MS_LOG(EXCEPTION) << op_name << ": The rank among 'query', 'key' and 'value' must be the same, but got "
                      << query_shape.size() << ", " << key_shape.size() << " and " << value_shape.size();
  }
  if (input_layout == FASInputLayoutMode::BSH) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBSHRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFlashAttentionScoreQueryBSHRank
                        << ", but got " << query_shape.size() << " and " << key_shape.size();
    }
    batch_size = query_shape[0];
    q_seq_len = query_shape[1];
    auto q_hidden_size = query_shape[2];
    if (q_hidden_size % q_head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << q_hidden_size
                        << " and " << q_head_num;
    }
    int64_t head_size = q_hidden_size / q_head_num;
    kv_seq_len = key_shape[kIndex1];
    kv_head_num = key_shape[kIndex2] / head_size;
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBNSDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFlashAttentionScoreQueryBNSDRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    if (q_head_num != query_shape[kIndex1]) {
      MS_LOG(EXCEPTION) << op_name << ": query_shape[1] must be equal to attribute 'head_num', but got "
                        << query_shape[1] << " and " << q_head_num;
    }
    q_seq_len = query_shape[kIndex2];
    kv_seq_len = key_shape[kIndex2];
    kv_head_num = key_shape[kIndex1];
  } else if (input_layout == FASInputLayoutMode::SBH) {
    if (query_shape.size() != kInputFlashAttentionScoreQuerySBHRank || key_shape.size() != query_shape.size()) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' and 'key' must be "
                        << kInputFlashAttentionScoreQuerySBHRank << ", but got " << query_shape.size() << " and "
                        << key_shape.size();
    }
    batch_size = query_shape[1];
    q_seq_len = query_shape[0];
    auto q_hidden_size = query_shape[2];
    if (q_hidden_size % q_head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << q_hidden_size
                        << " and " << q_head_num;
    }
    int64_t head_size = q_hidden_size / q_head_num;
    kv_seq_len = key_shape[kIndex0];
    kv_head_num = key_shape[kIndex2] / head_size;
  } else if (input_layout == FASInputLayoutMode::BSND) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBSNDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFlashAttentionScoreQueryBSNDRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    if (q_head_num != query_shape[kIndex2]) {
      MS_LOG(EXCEPTION) << op_name << ": query_shape[2] must be equal to attribute 'head_num', but got "
                        << query_shape[kIndex2] << " and " << q_head_num;
    }
    q_seq_len = query_shape[kIndex1];
    kv_seq_len = key_shape[kIndex1];
    kv_head_num = key_shape[kIndex2];
  } else {
    MS_LOG(EXCEPTION) << op_name << " support input layout: BSH, BNSD, SBH, BSND, TND.";
  }
  if (q_head_num % kv_head_num != 0) {
    MS_LOG(EXCEPTION) << op_name << ": The head num of 'key' must be a factor of the head num of 'query', but got "
                      << kv_head_num << " and " << q_head_num;
  }
  return std::vector<int64_t>{batch_size, q_seq_len, kv_seq_len};
}

ShapeArray FlashAttentionScoreFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto query_shape = input_infos[kFlashAttentionScoreInputQueryIndex]->GetShape();
  auto key_shape = input_infos[kFlashAttentionScoreInputKeyIndex]->GetShape();
  auto value_shape = input_infos[kFlashAttentionScoreInputValueIndex]->GetShape();
  ShapeVector dyn_rank{abstract::Shape::kShapeRankAny};
  if (IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputLayoutIndex])) {
    return ConstructInferShape(dyn_rank, query_shape, key_shape, value_shape);
  }
  auto input_layout_opt = input_infos[kFlashAttentionScoreInputLayoutIndex]->GetScalarValue<int64_t>();
  if (!input_layout_opt.has_value()) {
    return ConstructInferShape(dyn_rank, query_shape, key_shape, value_shape);
  }

  bool head_num_no_value = false;
  std::optional<int64_t> head_num_opt_cached;
  if (IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputHeadNumIndex])) {
    head_num_no_value = true;
  } else {
    head_num_opt_cached = input_infos[kFlashAttentionScoreInputHeadNumIndex]->GetScalarValue<int64_t>();
    if (!head_num_opt_cached.has_value()) {
      head_num_no_value = true;
    }
  }

  auto input_layout = input_layout_opt.value();
  if (input_layout == FASInputLayoutMode::TND || input_layout == FASInputLayoutMode::TH) {
    if (input_layout == FASInputLayoutMode::TND) {
      if (IsDynamicRank(query_shape)) {
        return ConstructInferShape(
          ShapeVector{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, kFlashAttentionScoreSoftmaxLastDim},
          query_shape, key_shape, value_shape, input_layout_opt);
      }
      return ConstructInferShape(ShapeVector{query_shape[0], query_shape[1], kFlashAttentionScoreSoftmaxLastDim},
                                 query_shape, key_shape, value_shape, input_layout_opt);
    } else {
      if (IsDynamicRank(query_shape)) {
        return ConstructInferShape(ShapeVector{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny},
                                   query_shape, key_shape, value_shape, input_layout_opt);
      }

      if (!head_num_opt_cached.has_value()) {
        MS_LOG(EXCEPTION) << op_name << ": 'head_num' must be provided when input layout is TH.";
      }
      int64_t q_head_num = head_num_opt_cached.value();
      q_head_num *= static_cast<int64_t>(kFlashAttentionScoreSoftmaxLastDim);
      return ConstructInferShape(ShapeVector{query_shape[0], q_head_num}, query_shape, key_shape, value_shape,
                                 input_layout_opt);
    }
  }

  if (IsDynamicRank(query_shape)) {
    return ConstructInferShape(ShapeVector{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                                           abstract::Shape::kShapeDimAny, kFlashAttentionScoreSoftmaxLastDim},
                               query_shape, key_shape, value_shape, input_layout_opt);
  }

  size_t seq_index = kIndex1, batch_index = kIndex0;
  if (input_layout == FASInputLayoutMode::SBH) {
    seq_index = kIndex0;
    batch_index = kIndex1;
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    seq_index = kIndex2;
  }
  if (head_num_no_value) {
    return ConstructInferShape(ShapeVector{query_shape[batch_index], abstract::Shape::kShapeDimAny,
                                           query_shape[seq_index], kFlashAttentionScoreSoftmaxLastDim},
                               query_shape, key_shape, value_shape, input_layout_opt);
  }

  auto q_head_num = head_num_opt_cached.value();
  if (IsDynamicShape(query_shape) || IsDynamic(key_shape)) {
    return ConstructInferShape(
      ShapeVector{query_shape[batch_index], q_head_num, query_shape[seq_index], kFlashAttentionScoreSoftmaxLastDim},
      query_shape, key_shape, value_shape, input_layout_opt);
  }

  auto shape_info = GetFASInfoFromInputLayout(input_layout, q_head_num, op_name, query_shape, key_shape, value_shape);

  CheckFlashAttentionScoreSparseMode(primitive, input_infos, shape_info, q_head_num);

  int64_t batch_size = shape_info[kIndex0];
  int64_t q_seq_len = shape_info[kIndex1];
  return ConstructInferShape(ShapeVector{batch_size, q_head_num, q_seq_len, kFlashAttentionScoreSoftmaxLastDim},
                             query_shape, key_shape, value_shape, input_layout_opt);
}

std::vector<TypeId> FlashAttentionScoreFuncImpl::InferType(const PrimitivePtr &prim,
                                                           const InferInfoPtrList &input_infos) const {
  auto op_name = prim->name();
  ValidateKeepProbAndDropMask(input_infos, op_name);

  const auto q_type = input_infos[kFlashAttentionScoreInputQueryIndex]->GetType();
  std::vector<TypeId> outs(kFlashAttentionScoreOutputsNum);
  outs[kFlashAttentionScoreOutputSoftmaxMaxIndex] = kNumberTypeFloat32;
  outs[kFlashAttentionScoreOutputSoftmaxSumIndex] = kNumberTypeFloat32;
  outs[kFlashAttentionScoreOutputSoftmaxOutIndex] = q_type;
  outs[kFlashAttentionScoreOutputAttentionOutIndex] = q_type;
  return outs;
}
}  // namespace ops
}  // namespace mindspore
