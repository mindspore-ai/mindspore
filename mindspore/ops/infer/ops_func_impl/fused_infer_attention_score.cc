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

#include "infer/ops_func_impl/fused_infer_attention_score.h"

#include <memory>
#include <string>

#include "include/op_enum.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/common_infer_fns.h"

namespace mindspore {
namespace ops {
void CheckActualSeqLengthsKvInPageAttention(int64_t query_seq_lengths, const std::string &prim_name,
                                            const std::vector<AbstractBasePtr> &input_args) {
  if (query_seq_lengths != 1) {
    // not in IFA branch, skip
    return;
  }
  const auto block_table = input_args[kFusedInferAttentionScoreInputBlockTableIndex];
  if (MS_LIKELY(IsOptionalInputNone(block_table))) {
    // page attention is not enabled, skip
    return;
  }
  const auto actual_seq_lengths_kv = input_args[kFusedInferAttentionScoreInputActualSeqLengthsKvIndex];
  if (IsOptionalInputNone(actual_seq_lengths_kv)) {
    MS_LOG(EXCEPTION) << "For " << prim_name
                      << ", actual_seq_lengths_kv cannot be none in PageAttention scenario when Q_S is 1.";
  }
}

BaseShapePtr FusedInferAttentionScoreFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto query_shape_vec = input_args[kFusedInferAttentionScoreInputQueryIndex]->GetShape()->GetShapeVector();
  BaseShapePtr attention_out_shape = input_args[kFusedInferAttentionScoreInputQueryIndex]->GetShape()->Clone();
  BaseShapePtr softmax_lse_shape = std::make_shared<abstract::TensorShape>(ShapeVector{1});
  if (IsDynamicRank(query_shape_vec)) {
    ShapeVector dyrank_shape{abstract::TensorShape::kShapeRankAny};
    attention_out_shape = std::make_shared<abstract::TensorShape>(dyrank_shape);
    softmax_lse_shape = std::make_shared<abstract::TensorShape>(dyrank_shape);
    return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({attention_out_shape, softmax_lse_shape}));
  }
  auto Batch = query_shape_vec[kIndex0];

  // get num_heads and input_layout
  auto head_num_value = input_args[kFusedInferAttentionScoreInputNumHeadsIndex]->GetValue();
  auto head_num_opt = GetScalarValue<int64_t>(head_num_value);
  auto input_layout_value = input_args[kFusedInferAttentionScoreInputLayoutIndex]->GetValue();
  auto input_layout_opt = GetScalarValue<int64_t>(input_layout_value);
  if (!head_num_opt.has_value() || !input_layout_opt.has_value()) {
    ShapeVector dyrank_shape{abstract::TensorShape::kShapeRankAny};
    attention_out_shape = std::make_shared<abstract::TensorShape>(dyrank_shape);
    softmax_lse_shape = std::make_shared<abstract::TensorShape>(dyrank_shape);
    return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({attention_out_shape, softmax_lse_shape}));
  }
  auto N = head_num_opt.value();
  auto input_layout = input_layout_opt.value();

  int64_t Q_S = 1;
  switch (input_layout) {
    case FASInputLayoutMode::BSH:
      Q_S = query_shape_vec[kIndex1];
      break;
    case FASInputLayoutMode::BNSD:
    case FASInputLayoutMode::BNSD_BSND:
      Q_S = query_shape_vec[kIndex2];
      break;
    case FASInputLayoutMode::BSND:
      Q_S = query_shape_vec[kIndex1];
      break;
    default:
      MS_LOG(EXCEPTION)
        << "For FusedInferAttentionScore, the input_layout should be one of 'BSH' 'BSND' 'BNSD' 'BNSD_BSND'.";
  }
  // In FIA PageAttention scenario, actual_seq_lengths_kv cannot be empty. When it is empty, aclnn incorrectly
  // reports actual_seq_lengths_kv as actual_seq_lengths, which can easily lead to user misunderstanding.
  // Therefore, explicitly intercept here.
  CheckActualSeqLengthsKvInPageAttention(Q_S, primitive->name(), input_args);

  if (input_layout == FASInputLayoutMode::BNSD_BSND) {
    // swap the N-axis and S-axis for the shape of attention_out
    int64_t Q_D = query_shape_vec.at(kIndex3);
    ShapeVector out_shape{Batch, Q_S, N, Q_D};
    attention_out_shape = std::make_shared<abstract::TensorShape>(out_shape);
  }
  const auto softmax_lse_flag_value = input_args[kFusedInferAttentionScoreInputSoftmaxLseFlagIndex]->GetValue();
  const auto softmax_lse_flag = GetScalarValue<bool>(softmax_lse_flag_value).value_or(false);
  if (softmax_lse_flag) {
    ShapeVector softmax_shape{Batch, N, Q_S, 1};
    softmax_lse_shape = std::make_shared<abstract::TensorShape>(softmax_shape);
  }
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({attention_out_shape, softmax_lse_shape}));
}

TypePtr FusedInferAttentionScoreFuncImpl::InferType(const PrimitivePtr &prim,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto query_type = input_args[kFusedInferAttentionScoreInputQueryIndex]->GetType();
  auto query_type_id = query_type->cast<TensorTypePtr>()->element()->type_id();

  auto attention_out_type = query_type;
  auto softmax_lse_type = std::make_shared<TensorType>(kFloat32);
  bool has_deqScale1 = !input_args[kFusedInferAttentionScoreInputDequantScale1Index]->GetType()->isa<TypeNone>();
  bool has_qScale1 = !input_args[kFusedInferAttentionScoreInputQuantScale1Index]->GetType()->isa<TypeNone>();
  bool has_deqScale2 = !input_args[kFusedInferAttentionScoreInputDequantScale2Index]->GetType()->isa<TypeNone>();
  bool has_qScale2 = !input_args[kFusedInferAttentionScoreInputQuantScale2Index]->GetType()->isa<TypeNone>();
  bool has_qOffset2 = !input_args[kFusedInferAttentionScoreInputQuantOffset2Index]->GetType()->isa<TypeNone>();
  if (query_type_id == TypeId::kNumberTypeInt8) {
    if (has_deqScale1 && has_qScale1 && has_deqScale2 && !has_qScale2 && !has_qOffset2) {
      attention_out_type = std::make_shared<TensorType>(kFloat16);
    }
  } else {
    attention_out_type = has_qScale2 ? std::make_shared<TensorType>(kInt8) : query_type;
  }
  return std::make_shared<Tuple>(std::vector<TypePtr>{attention_out_type, softmax_lse_type});
}

std::set<int64_t> FusedInferAttentionScoreFuncImpl::GetValueDependArgIndices() const {
  // The aclnn kernelmod depends on the values of the elements at these three index positions,
  // which will be used for tensor-to-vector conversion.
  return {kFusedInferAttentionScoreInputActualSeqLengthsIndex, kFusedInferAttentionScoreInputActualSeqLengthsKvIndex,
          kFusedInferAttentionScoreInputActualSharedPrefixLenIndex};
}
}  // namespace ops
}  // namespace mindspore
