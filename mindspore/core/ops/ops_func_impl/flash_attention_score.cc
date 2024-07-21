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

#include "ops/ops_func_impl/flash_attention_score.h"

#include <string>
#include <map>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "ops/op_enum.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
constexpr size_t kFlashAttentionScoreSoftmaxLastDim = 8;
constexpr size_t kInputFlashAttentionScoreQueryBSHRank = 3;
constexpr size_t kInputFlashAttentionScoreQuerySBHRank = 3;
constexpr size_t kInputFlashAttentionScoreQueryTNDRank = 3;
constexpr size_t kInputFlashAttentionScoreQueryBNSDRank = 4;
constexpr size_t kInputFlashAttentionScoreQueryBSNDRank = 4;
constexpr size_t kFAGRealShiftCompressionDim = 1024;
constexpr size_t kInputFlashAttentionScoreAttnMaskCompressionDim = 2048;
constexpr auto kEnableRingAttention = "enable_ring_attention";
constexpr auto kEnableFlashSP = "enable_flash_sp";
constexpr auto kEnableRASendRecv = "enable_ra_send_recv";

// None indicates that the optional input is not passed
bool IsFlashAttentionScoreOptionalInputNotPass(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->GetType()->type_id() == kMetaTypeNone;
}

void CheckFlashAttentionScoreInputShape(const AbstractBasePtr &input, const ShapeVector &expect_shape,
                                        const std::string &op_name, const std::string &input_name,
                                        bool optional = false) {
  MS_EXCEPTION_IF_NULL(input);
  if (IsFlashAttentionScoreOptionalInputNotPass(input) && optional) {
    return;
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input->GetShape())[kShape];
  if (input_shape != expect_shape) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be " << expect_shape
                      << ", but got shape is " << input_shape;
  }
}

void CheckFlashAttentionScoreInputShape(const AbstractBasePtr &input, const std::vector<ShapeVector> &expect_shape_list,
                                        const std::string &op_name, const std::string &input_name,
                                        bool optional = false) {
  MS_EXCEPTION_IF_NULL(input);
  if (IsFlashAttentionScoreOptionalInputNotPass(input) && optional) {
    return;
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input->GetShape())[kShape];
  if (std::all_of(expect_shape_list.begin(), expect_shape_list.end(),
                  [&input_shape](const ShapeVector &expect_shape) { return input_shape != expect_shape; })) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input " << input_name << " must be one of " << expect_shape_list
                      << ", but got shape is " << input_shape;
  }
}

void CheckFlashAttentionScoreAttnMaskShape(const AbstractBasePtr &attn_mask, const std::string &op_name,
                                           int64_t sparse_mode, int64_t batch_size, int64_t q_head_num,
                                           int64_t q_seq_len, int64_t kv_seq_len) {
  const std::vector<int64_t> need_compress_attn_mask_mode = {
    kSparseLeftUpCausal, kSparseRightDownCausal, kSparseBand,      kSparsePrefix,
    kSparseGlobal,       kSparseDilated,         kSparseBlockLocal};
  if (std::find(need_compress_attn_mask_mode.begin(), need_compress_attn_mask_mode.end(), sparse_mode) !=
      need_compress_attn_mask_mode.end()) {
    CheckFlashAttentionScoreInputShape(
      attn_mask, {kInputFlashAttentionScoreAttnMaskCompressionDim, kInputFlashAttentionScoreAttnMaskCompressionDim},
      op_name, "attn_mask");
  } else {
    auto is_attn_mask_optional = sparse_mode == kSparseDefaultMask;
    CheckFlashAttentionScoreInputShape(attn_mask,
                                       {{batch_size, q_head_num, q_seq_len, kv_seq_len},
                                        {batch_size, 1, q_seq_len, kv_seq_len},
                                        {q_seq_len, kv_seq_len}},
                                       op_name, "attn_mask", is_attn_mask_optional);
  }
}

void CheckFlashAttentionScorePrefix(const AbstractBasePtr &prefix, const std::string &op_name, int64_t sparse_mode,
                                    int64_t batch_size) {
  if (sparse_mode == kSparsePrefix) {
    auto prefix_type = prefix->GetType();
    MS_EXCEPTION_IF_NULL(prefix_type);
    if (!prefix_type->isa<Tuple>()) {
      MS_LOG(EXCEPTION) << "For [" << op_name << "], prefix type should be TupleType.";
    }
    auto prefix_tuple = prefix_type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(prefix_tuple);
    if (prefix_tuple->elements().size() != LongToSize(batch_size)) {
      MS_LOG(EXCEPTION) << "For [" << op_name << "], prefix list size should be equal to " << batch_size << ", but got "
                        << prefix_tuple->elements().size();
    }
    for (const auto &element : prefix_tuple->elements()) {
      if (element->type_id() != kNumberTypeInt64) {
        MS_LOG(EXCEPTION) << "For [" << op_name << "], prefix element type should be int64.";
      }
    }
  } else {
    if (!IsFlashAttentionScoreOptionalInputNotPass(prefix)) {
      MS_LOG(EXCEPTION) << op_name << ": 'prefix' must be None if sparse_mode is not " << kSparsePrefix;
    }
  }
}

void CheckFlashAttentionScoreSparseMode(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                        const std::vector<int64_t> &shape_info, int64_t q_head_num) {
  auto op_name = primitive->name();
  int64_t batch_size = shape_info[kIndex0];
  int64_t q_seq_len = shape_info[kIndex1];
  int64_t kv_seq_len = shape_info[kIndex2];
  auto sparse_mode_opt = GetScalarValue<int64_t>(input_args[kFlashAttentionScoreInputSparseModeIndex]->GetValue());
  if (sparse_mode_opt.has_value()) {
    auto sparse_mode = sparse_mode_opt.value();

    bool enable_ring_attention = false;
    if (primitive->HasAttr(kEnableRingAttention)) {
      auto enable_ring_attention_valueptr = primitive->GetAttr(kEnableRingAttention);
      if (enable_ring_attention_valueptr->isa<BoolImm>()) {
        enable_ring_attention = enable_ring_attention_valueptr->cast<BoolImmPtr>()->value();
      } else {
        MS_LOG(EXCEPTION) << "enable_ring_attention should be bool";
      }
    }
    if (primitive->HasAttr(kEnableRASendRecv)) {
      auto enable_ra_sendrecv_valueptr = primitive->GetAttr(kEnableRASendRecv);
      if (!(enable_ra_sendrecv_valueptr->isa<BoolImm>())) {
        MS_LOG(EXCEPTION) << "enable_ra_sendrecv should be bool";
      }
    }
    bool enable_flash_sp = false;
    if (primitive->HasAttr(kEnableFlashSP)) {
      auto enable_flash_sp_valueptr = primitive->GetAttr(kEnableFlashSP);
      if (enable_flash_sp_valueptr->isa<BoolImm>()) {
        enable_flash_sp = enable_flash_sp_valueptr->cast<BoolImmPtr>()->value();
      } else {
        MS_LOG(ERROR) << "enable_flash_sp should be bool";
      }
    }
    if ((!enable_ring_attention && !enable_flash_sp) ||
        !IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputAttnMaskIndex])) {
      CheckFlashAttentionScoreAttnMaskShape(input_args[kFlashAttentionScoreInputAttnMaskIndex], op_name, sparse_mode,
                                            batch_size, q_head_num, q_seq_len, kv_seq_len);
    }
    CheckFlashAttentionScorePrefix(input_args[kFlashAttentionScoreInputPrefixIndex], op_name, sparse_mode, batch_size);
  }
}

BaseShapePtr ConstructInferShape(const ShapeVector &softmax_shape, const ShapeVector &query_shape) {
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList(
    {std::make_shared<abstract::Shape>(softmax_shape), std::make_shared<abstract::Shape>(softmax_shape),
     std::make_shared<abstract::Shape>(ShapeVector{1}), std::make_shared<abstract::Shape>(query_shape)}));
}

std::vector<int64_t> GetFASInfoFromInputLayout(int64_t input_layout, int64_t q_head_num, const std::string &op_name,
                                               const ShapeVector &query_shape, const ShapeVector &key_shape) {
  int64_t batch_size = -1;
  int64_t q_seq_len = -1;
  int64_t kv_seq_len = -1;
  int64_t kv_head_num = -1;
  if (input_layout == FASInputLayoutMode::BSH) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBSHRank || key_shape.size() != query_shape.size()) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' and 'key' must be "
                        << kInputFlashAttentionScoreQueryBSHRank << ", but got " << query_shape.size() << " and "
                        << key_shape.size();
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

BaseShapePtr FlashAttentionScoreFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto query_shape = input_args[kFlashAttentionScoreInputQueryIndex]->GetShape()->GetShapeVector();
  auto key_shape = input_args[kFlashAttentionScoreInputKeyIndex]->GetShape()->GetShapeVector();
  ShapeVector dyn_rank{abstract::Shape::kShapeRankAny};
  ShapeVector dyn_shape{abstract::Shape::kShapeDimAny};
  if (IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputLayoutIndex])) {
    return ConstructInferShape(dyn_rank, query_shape);
  }
  auto input_layout_value = input_args[kFlashAttentionScoreInputLayoutIndex]->GetValue();
  MS_EXCEPTION_IF_NULL(input_layout_value);
  auto input_layout_opt = GetScalarValue<int64_t>(input_layout_value);
  if (!input_layout_opt.has_value()) {
    return ConstructInferShape(dyn_rank, query_shape);
  }
  auto input_layout = input_layout_opt.value();
  if (input_layout == FASInputLayoutMode::TND) {
    if (IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputActualSeqQlenIndex]) ||
        IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputActualSeqKVlenIndex])) {
      MS_LOG(EXCEPTION) << op_name << ": actual_seq_qlen and actual_seq_kvlen should be not none.";
    }
    if (IsDynamicRank(query_shape)) {
      return ConstructInferShape(
        ShapeVector{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, kFlashAttentionScoreSoftmaxLastDim},
        query_shape);
    }
    return ConstructInferShape(ShapeVector{query_shape[0], query_shape[1], kFlashAttentionScoreSoftmaxLastDim},
                               query_shape);
  }

  if (IsDynamicRank(query_shape)) {
    return ConstructInferShape(ShapeVector{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                                           abstract::Shape::kShapeDimAny, kFlashAttentionScoreSoftmaxLastDim},
                               query_shape);
  }

  auto head_num_value = input_args[kFlashAttentionScoreInputHeadNumIndex]->GetValue();
  MS_EXCEPTION_IF_NULL(head_num_value);
  bool head_num_no_value = false;
  if (IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputHeadNumIndex])) {
    head_num_no_value = true;
  } else {
    auto head_opt = GetScalarValue<int64_t>(head_num_value);
    if (!head_opt.has_value()) {
      head_num_no_value = true;
    }
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
                               query_shape);
  }

  auto head_num_opt = GetScalarValue<int64_t>(head_num_value);
  auto q_head_num = head_num_opt.value();
  if (IsDynamicShape(query_shape) || IsDynamic(key_shape)) {
    return ConstructInferShape(
      ShapeVector{query_shape[batch_index], q_head_num, query_shape[seq_index], kFlashAttentionScoreSoftmaxLastDim},
      query_shape);
  }

  auto shape_info = GetFASInfoFromInputLayout(input_layout, q_head_num, op_name, query_shape, key_shape);
  int64_t batch_size = shape_info[kIndex0];
  int64_t q_seq_len = shape_info[kIndex1];
  int64_t kv_seq_len = shape_info[kIndex2];

  CheckFlashAttentionScoreInputShape(input_args[kFlashAttentionScoreInputValueIndex], key_shape, op_name, "value");
  CheckFlashAttentionScoreInputShape(input_args[kFlashAttentionScoreInputRealShiftIndex],
                                     {{batch_size, q_head_num, q_seq_len, kv_seq_len},
                                      {1, q_head_num, q_seq_len, kv_seq_len},
                                      {batch_size, q_head_num, kFAGRealShiftCompressionDim, kv_seq_len},
                                      {1, q_head_num, kFAGRealShiftCompressionDim, kv_seq_len}},
                                     op_name, "real_shift", true);
  CheckFlashAttentionScoreInputShape(input_args[kFlashAttentionScoreInputDropMaskIndex],
                                     {batch_size, q_head_num, q_seq_len, kv_seq_len / 8}, op_name, "drop_mask", true);
  CheckFlashAttentionScoreSparseMode(primitive, input_args, shape_info, q_head_num);

  return ConstructInferShape(ShapeVector{batch_size, q_head_num, q_seq_len, kFlashAttentionScoreSoftmaxLastDim},
                             query_shape);
}

TypePtr FlashAttentionScoreFuncImpl::InferType(const PrimitivePtr &prim,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  const std::set valid_types = {kFloat16, kBFloat16};
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;

  (void)types.emplace("query", input_args[kFlashAttentionScoreInputQueryIndex]->GetType());
  (void)types.emplace("key", input_args[kFlashAttentionScoreInputKeyIndex]->GetType());
  (void)types.emplace("value", input_args[kFlashAttentionScoreInputValueIndex]->GetType());
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputRealShiftIndex])) {
    (void)types.emplace("real_shift", input_args[kFlashAttentionScoreInputRealShiftIndex]->GetType());
  }
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputPaddingMaskIndex])) {
    MS_LOG(EXCEPTION) << op_name << ": 'padding_mask' must be None currently.";
  }
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputAttnMaskIndex])) {
    auto attn_mask_type = input_args[kFlashAttentionScoreInputAttnMaskIndex]->GetType();
    std::set attn_mask_valid_types = {kUInt8, kBool};
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ms_context->IsEnableInferBoost()) {
      (void)attn_mask_valid_types.emplace(kFloat16);
    }
    CheckAndConvertUtils::CheckTensorTypeValid("attn_mask", attn_mask_type, attn_mask_valid_types, op_name);
  }

  auto keep_prob_value_ptr = input_args[kFlashAttentionScoreInputKeepProbIndex]->GetValue();
  MS_EXCEPTION_IF_NULL(keep_prob_value_ptr);
  auto keep_prob_opt = GetScalarValue<float>(keep_prob_value_ptr);
  if (keep_prob_opt.has_value()) {
    // check keep_prob
    auto keep_prob = keep_prob_opt.value();
    if (keep_prob > 1 || keep_prob < 0) {
      MS_LOG(EXCEPTION) << op_name << ": attribute `keep_prob` must be a floating point number in [0, 1], but got "
                        << keep_prob;
    }
    if (common::IsFloatEqual(keep_prob, 1.0)) {
      if (!IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputDropMaskIndex])) {
        MS_LOG(EXCEPTION) << op_name << ": 'drop_mask' must be None when keep_prob is 1.0.";
      }
    } else {
      auto drop_mask_type = input_args[kFlashAttentionScoreInputDropMaskIndex]->GetType();
      CheckAndConvertUtils::CheckTensorTypeValid("drop_mask", drop_mask_type, {kUInt8}, op_name);
    }
  }

  TypePtrList output_type_ptr_list(kFlashAttentionScoreOutputsNum);
  output_type_ptr_list[kFlashAttentionScoreOutputSoftmaxMaxIndex] = std::make_shared<TensorType>(kFloat32);
  output_type_ptr_list[kFlashAttentionScoreOutputSoftmaxSumIndex] = std::make_shared<TensorType>(kFloat32);
  output_type_ptr_list[kFlashAttentionScoreOutputSoftmaxOutIndex] = std::make_shared<TensorType>(type);
  output_type_ptr_list[kFlashAttentionScoreOutputAttentionOutIndex] = std::make_shared<TensorType>(type);
  return std::make_shared<Tuple>(output_type_ptr_list);
}
}  // namespace ops
}  // namespace mindspore
