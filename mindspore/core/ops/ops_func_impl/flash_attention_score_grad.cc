/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/flash_attention_score_grad.h"

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
namespace {
enum FASGradInputIndex : size_t {
  kFASGradInputQueryIndex = 0,
  kFASGradInputKeyIndex,
  kFASGradInputValueIndex,
  kFASGradInputDyIndex,
  kFASGradInputPseShiftIndex,
  kFASGradInputDropMaskIndex,
  kFASGradInputPaddingMaskIndex,
  kFASGradInputAttnMaskIndex,
  kFASGradInputSoftmaxMaxIndex,
  kFASGradInputSoftmaxSumIndex,
  kFASGradInputSoftmaxOutIndex,
  kFASGradInputAttentionInIndex,
  kFASGradInputPrefixIndex,
  kFASGradInputActualSeqQlenIndex,
  kFASGradInputActualSeqKVlenIndex,
  kFASGradInputHeadNumIndex,
  kFASGradInputKeepProbIndex,
  kFASGradInputScaleValueIndex,
  kFASGradInputPreTokensIndex,
  kFASGradInputNextTokensIndex,
  kFASGradInputInnerPreciseIndex,
  kFASGradInputLayoutIndex,
  kFASGradInputSparseModeIndex,
  kFASGradInputsNum,
};
enum FASGradOutputIndex : size_t {
  kFASGradOutputDqIndex = 0,
  kFASGradOutputDkIndex,
  kFASGradOutputDvIndex,
  kFASGradOutputDpseIndex,
  kFASGradOutputsNum,
};
enum FASGradSparseMode : int64_t {
  kFAGSparseDefaultMask = 0,
  kFAGSparseAllMask,
  kFAGSparseLeftUpCausal,
  kFAGSparseRightDownCausal,
  kFAGSparseBand,
  kFAGSparsePrefix,
  kFAGSparseGlobal,
  kFAGSparseDilated,
  kFAGSparseBlockLocal,
};
}  // namespace

constexpr size_t kFASGradSoftmaxLastDim = 8;
constexpr size_t kInputFASGradQueryBSHRank = 3;
constexpr size_t kInputFASGradQuerySBHRank = 3;
constexpr size_t kInputFASGradQueryTNDRank = 3;
constexpr size_t kInputFASGradQueryBNSDRank = 4;
constexpr size_t kInputFASGradQueryBSNDRank = 4;
constexpr size_t kInputFASGradAttnMaskCompressionDim = 2048;
constexpr size_t kFARealShiftCompressionDim = 1024;
// None indicates that the optional input is not passed
bool IsFlashAttentionScoreGradOptionalInputNotPass(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->GetType()->type_id() == kMetaTypeNone;
}

void CheckFlashAttentionScoreGradInputShape(const AbstractBasePtr &input, const ShapeVector &expect_shape,
                                            const std::string &op_name, const std::string &input_name,
                                            bool optional = false) {
  MS_EXCEPTION_IF_NULL(input);
  if (IsFlashAttentionScoreGradOptionalInputNotPass(input) && optional) {
    return;
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input->GetShape())[kShape];
  if (input_shape != expect_shape) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be " << expect_shape
                      << ", but got shape is " << input_shape;
  }
}

void CheckFlashAttentionScoreGradInputShape(const AbstractBasePtr &input,
                                            const std::vector<ShapeVector> &expect_shape_list,
                                            const std::string &op_name, const std::string &input_name,
                                            bool optional = false) {
  MS_EXCEPTION_IF_NULL(input);
  if (IsFlashAttentionScoreGradOptionalInputNotPass(input) && optional) {
    return;
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input->GetShape())[kShape];
  if (std::all_of(expect_shape_list.begin(), expect_shape_list.end(),
                  [&input_shape](const ShapeVector &expect_shape) { return input_shape != expect_shape; })) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be one of " << expect_shape_list
                      << ", but got shape is " << input_shape;
  }
}

void CheckFlashAttentionScoreGradAttnMaskShape(const AbstractBasePtr &attn_mask, const std::string &op_name,
                                               int64_t sparse_mode, int64_t batch_size, int64_t q_head_num,
                                               int64_t q_seq_len, int64_t kv_seq_len) {
  const std::vector<int64_t> need_compress_attn_mask_mode = {kFAGSparseLeftUpCausal, kFAGSparseRightDownCausal,
                                                             kFAGSparseBand};
  if (std::find(need_compress_attn_mask_mode.begin(), need_compress_attn_mask_mode.end(), sparse_mode) !=
      need_compress_attn_mask_mode.end()) {
    CheckFlashAttentionScoreGradInputShape(
      attn_mask, {kInputFASGradAttnMaskCompressionDim, kInputFASGradAttnMaskCompressionDim}, op_name, "attn_mask");
  } else {
    auto is_attn_mask_optional = sparse_mode == kFAGSparseDefaultMask;
    CheckFlashAttentionScoreGradInputShape(attn_mask,
                                           {{batch_size, q_head_num, q_seq_len, kv_seq_len},
                                            {batch_size, 1, q_seq_len, kv_seq_len},
                                            {q_seq_len, kv_seq_len}},
                                           op_name, "attn_mask", is_attn_mask_optional);
  }
}

void CheckFlashAttentionScoreGradPrefix(const AbstractBasePtr &prefix, const std::string &op_name, int64_t sparse_mode,
                                        int64_t batch_size) {
  if (sparse_mode == kFAGSparsePrefix) {
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
    if (!IsFlashAttentionScoreGradOptionalInputNotPass(prefix)) {
      MS_LOG(EXCEPTION) << op_name << ": 'prefix' must be None if sparse_mode is not " << kFAGSparsePrefix;
    }
  }
}

std::vector<int64_t> GetFASGradInfoFromInputLayout(int64_t input_layout, int64_t q_head_num, const std::string &op_name,
                                                   const ShapeVector &query_shape, const ShapeVector &key_shape) {
  int64_t batch_size = -1;
  int64_t q_seq_len = -1;
  int64_t kv_seq_len = -1;
  int64_t kv_head_num = -1;
  if (input_layout == FASInputLayoutMode::BSH) {
    if (query_shape.size() != kInputFASGradQueryBSHRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of input `query` must be " << kInputFASGradQueryBSHRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    q_seq_len = query_shape[kIndex1];
    auto q_hidden_size = query_shape[kIndex2];
    if (q_hidden_size % q_head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << q_hidden_size
                        << " and " << q_head_num;
    }
    int64_t head_size = q_hidden_size / q_head_num;
    kv_seq_len = key_shape[kIndex1];
    kv_head_num = key_shape[kIndex2] / head_size;
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    if (query_shape.size() != kInputFASGradQueryBNSDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFASGradQueryBNSDRank << ", but got "
                        << query_shape.size();
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
    if (query_shape.size() != kInputFASGradQuerySBHRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of input `query` must be " << kInputFASGradQuerySBHRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex1];
    q_seq_len = query_shape[kIndex0];
    auto q_hidden_size = query_shape[kIndex2];
    if (q_hidden_size % q_head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << q_hidden_size
                        << " and " << q_head_num;
    }
    int64_t head_size = q_hidden_size / q_head_num;
    kv_seq_len = key_shape[kIndex0];
    kv_head_num = key_shape[kIndex2] / head_size;
  } else if (input_layout == FASInputLayoutMode::BSND) {
    if (query_shape.size() != kInputFASGradQueryBSNDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFASGradQueryBSNDRank << ", but got "
                        << query_shape.size();
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
    MS_LOG(EXCEPTION) << op_name << ": The head num of key must be a factor of the head num of query, but got "
                      << kv_head_num << " and " << q_head_num;
  }
  return std::vector<int64_t>{batch_size, q_seq_len, kv_seq_len};
}

BaseShapePtr FlashAttentionScoreGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto query_shape = input_args[kFASGradInputQueryIndex]->GetShape()->GetShapeVector();
  auto key_shape = input_args[kFASGradInputKeyIndex]->GetShape()->GetShapeVector();

  abstract::BaseShapePtrList output_shape_ptr_list(kFASGradOutputsNum);
  output_shape_ptr_list[kFASGradOutputDqIndex] = std::make_shared<abstract::Shape>(query_shape);
  output_shape_ptr_list[kFASGradOutputDkIndex] = std::make_shared<abstract::Shape>(key_shape);
  auto value_shape = input_args[kFASGradInputValueIndex]->GetShape()->GetShapeVector();
  output_shape_ptr_list[kFASGradOutputDvIndex] = std::make_shared<abstract::Shape>(value_shape);
  ShapeVector pse_shape{0};
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputPseShiftIndex])) {
    pse_shape = input_args[kFASGradInputPseShiftIndex]->GetShape()->GetShapeVector();
  }
  output_shape_ptr_list[kFASGradOutputDpseIndex] = std::make_shared<abstract::Shape>(pse_shape);

  auto input_layout_opt = GetScalarValue<int64_t>(input_args[kFASGradInputLayoutIndex]->GetValue());
  if (!input_layout_opt.has_value() || IsDynamic(query_shape) || IsDynamic(key_shape)) {
    return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
  }
  auto input_layout = input_layout_opt.value();
  if (input_layout == FASInputLayoutMode::TND) {
    if (IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputActualSeqQlenIndex]) ||
        IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputActualSeqKVlenIndex])) {
      MS_LOG(EXCEPTION) << op_name << ": actual_seq_qlen and actual_seq_kvlen should be not none.";
    }
    return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
  }

  auto q_head_num_opt = GetScalarValue<int64_t>(input_args[kFASGradInputHeadNumIndex]->GetValue());
  if (q_head_num_opt.has_value()) {
    // check shape
    auto q_head_num = q_head_num_opt.value();
    auto shape_info = GetFASGradInfoFromInputLayout(input_layout, q_head_num, op_name, query_shape, key_shape);
    int64_t batch_size = shape_info[kIndex0];
    int64_t q_seq_len = shape_info[kIndex1];
    int64_t kv_seq_len = shape_info[kIndex2];

    CheckFlashAttentionScoreGradInputShape(input_args[kFASGradInputDyIndex], query_shape, op_name, "dy");
    CheckFlashAttentionScoreGradInputShape(input_args[kFASGradInputAttentionInIndex], query_shape, op_name,
                                           "attention_in");
    CheckFlashAttentionScoreGradInputShape(input_args[kFASGradInputPseShiftIndex],
                                           {{batch_size, q_head_num, q_seq_len, kv_seq_len},
                                            {1, q_head_num, q_seq_len, kv_seq_len},
                                            {batch_size, q_head_num, kFARealShiftCompressionDim, kv_seq_len},
                                            {1, q_head_num, kFARealShiftCompressionDim, kv_seq_len}},
                                           op_name, "pse_shift", true);
    CheckFlashAttentionScoreGradInputShape(input_args[kFASGradInputDropMaskIndex],
                                           {batch_size, q_head_num, q_seq_len, kv_seq_len / 8}, op_name, "drop_mask",
                                           true);

    auto sparse_mode_opt = GetScalarValue<int64_t>(input_args[kFASGradInputSparseModeIndex]->GetValue());
    if (sparse_mode_opt.has_value()) {
      auto sparse_mode = sparse_mode_opt.value();
      CheckFlashAttentionScoreGradAttnMaskShape(input_args[kFASGradInputAttnMaskIndex], op_name, sparse_mode,
                                                batch_size, q_head_num, q_seq_len, kv_seq_len);
      CheckFlashAttentionScoreGradPrefix(input_args[kFASGradInputPrefixIndex], op_name, sparse_mode, batch_size);
    }

    CheckFlashAttentionScoreGradInputShape(input_args[kFASGradInputSoftmaxMaxIndex],
                                           {batch_size, q_head_num, q_seq_len, kFASGradSoftmaxLastDim}, op_name,
                                           "softmax_max");
    CheckFlashAttentionScoreGradInputShape(input_args[kFASGradInputSoftmaxSumIndex],
                                           {batch_size, q_head_num, q_seq_len, kFASGradSoftmaxLastDim}, op_name,
                                           "softmax_sum");
    CheckFlashAttentionScoreGradInputShape(input_args[kFASGradInputSoftmaxOutIndex], ShapeVector{1}, op_name,
                                           "softmax_out", true);
  }

  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TypePtr FlashAttentionScoreGradFuncImpl::InferType(const PrimitivePtr &prim,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types1;
  std::map<std::string, TypePtr> types2;
  // "x", "kernel_query", "kernel_key", "kernel_value", "gamma", " beta", "bias_query", "bias_key", "bias_value"
  (void)types1.emplace("query", input_args[kFASGradInputQueryIndex]->GetType());
  (void)types1.emplace("key", input_args[kFASGradInputKeyIndex]->GetType());
  (void)types1.emplace("value", input_args[kFASGradInputValueIndex]->GetType());
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputPseShiftIndex])) {
    (void)types1.emplace("pse_shift", input_args[kFASGradInputPseShiftIndex]->GetType());
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputAttnMaskIndex])) {
    auto attn_mask_type = input_args[kFASGradInputAttnMaskIndex]->GetType();
    CheckAndConvertUtils::CheckTensorTypeValid("attn_mask", attn_mask_type, {kUInt8, kBool}, op_name);
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputPaddingMaskIndex])) {
    MS_LOG(EXCEPTION) << op_name << ": 'padding_mask' must be None currently.";
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputAttentionInIndex])) {
    (void)types1.emplace("attention_in", input_args[kFASGradInputAttentionInIndex]->GetType());
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputSoftmaxMaxIndex])) {
    (void)types2.emplace("softmax_max", input_args[kFASGradInputSoftmaxMaxIndex]->GetType());
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputSoftmaxSumIndex])) {
    (void)types2.emplace("softmax_sum", input_args[kFASGradInputSoftmaxSumIndex]->GetType());
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputSoftmaxOutIndex])) {
    (void)types1.emplace("softmax_out", input_args[kFASGradInputSoftmaxOutIndex]->GetType());
  }
  (void)types1.emplace("dy", input_args[kFASGradInputDyIndex]->GetType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types1, {kFloat16, kBFloat16}, op_name);
  if (!types2.empty()) {
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types2, {kFloat32}, op_name);
  }

  auto keep_prob_value_ptr = input_args[kFASGradInputKeepProbIndex]->GetValue();
  MS_EXCEPTION_IF_NULL(keep_prob_value_ptr);
  auto keep_prob_opt = GetScalarValue<float>(keep_prob_value_ptr);
  if (keep_prob_opt.has_value()) {
    auto keep_prob = keep_prob_opt.value();
    if (keep_prob > 1 || keep_prob < 0) {
      MS_LOG(EXCEPTION) << op_name << ": attribute `keep_prob` must be a floating point number in [0, 1], but got "
                        << keep_prob;
    }
    if (common::IsFloatEqual(keep_prob, 1.0)) {
      if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputDropMaskIndex])) {
        MS_LOG(EXCEPTION) << op_name << ": 'drop_mask' must be None when keep_prob is 1.0.";
      }
    } else {
      if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_args[kFASGradInputDropMaskIndex])) {
        auto drop_mask_type = input_args[kFASGradInputDropMaskIndex]->GetType();
        CheckAndConvertUtils::CheckTensorTypeValid("drop_mask", drop_mask_type, {kUInt8}, op_name);
      }
    }
  }

  std::vector<TypePtr> output_type_ptr_list(kFASGradOutputsNum);
  output_type_ptr_list[kFASGradOutputDqIndex] = std::make_shared<TensorType>(type);
  output_type_ptr_list[kFASGradOutputDkIndex] = std::make_shared<TensorType>(type);
  output_type_ptr_list[kFASGradOutputDvIndex] = std::make_shared<TensorType>(type);
  output_type_ptr_list[kFASGradOutputDpseIndex] = std::make_shared<TensorType>(type);
  return std::make_shared<Tuple>(output_type_ptr_list);
}

}  // namespace ops
}  // namespace mindspore
