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

#include "ops/ops_func_impl/fused_infer_attention_score.h"

#include <memory>

#include "ops/op_enum.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FusedInferAttentionScoreFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto query_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  BaseShapePtr attention_out_shape = input_args[kInputIndex0]->GetShape()->Clone();
  BaseShapePtr softmax_lse_shape = attention_out_shape;
  if (IsDynamicRank(query_shape_vec)) {
    ShapeVector dyrank_shape{abstract::TensorShape::kShapeRankAny};
    attention_out_shape = std::make_shared<abstract::TensorShape>(dyrank_shape);
    softmax_lse_shape = std::make_shared<abstract::TensorShape>(dyrank_shape);
  } else {
    auto Batch = query_shape_vec[kIndex0];

    auto head_num_value = input_args[kFusedInferAttentionScoreInputNumHeadsIndex]->GetValue();
    MS_EXCEPTION_IF_NULL(head_num_value);
    auto head_num_opt = GetScalarValue<int64_t>(head_num_value);
    auto N = head_num_opt.value();

    auto input_layout_value = input_args[kFusedInferAttentionScoreInputLayoutIndex]->GetValue();
    MS_EXCEPTION_IF_NULL(input_layout_value);
    auto input_layout_opt = GetScalarValue<int64_t>(input_layout_value);
    if (!input_layout_opt.has_value()) {
      ShapeVector dyrank_shape{abstract::TensorShape::kShapeRankAny};
      attention_out_shape = std::make_shared<abstract::TensorShape>(dyrank_shape);
      softmax_lse_shape = std::make_shared<abstract::TensorShape>(dyrank_shape);
    } else {
      int64_t Q_S = 1;
      auto input_layout = input_layout_opt.value();
      switch (input_layout) {
        case FASInputLayoutMode::BSH:
          Q_S = query_shape_vec[kIndex1];
          break;
        case FASInputLayoutMode::BNSD:
          Q_S = query_shape_vec[kIndex2];
          break;
        case FASInputLayoutMode::BSND:
          Q_S = query_shape_vec[kIndex1];
          break;
        default:
          MS_LOG(EXCEPTION) << "For FusedInferAttentionScore, the input_layout should be one of 'BSH' 'BSND' 'BSND'.";
      }
      ShapeVector softmax_shape{Batch, N, Q_S, 1};
      softmax_lse_shape = std::make_shared<abstract::TensorShape>(softmax_shape);
    }
  }
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({attention_out_shape, softmax_lse_shape}));
}

TypePtr FusedInferAttentionScoreFuncImpl::InferType(const PrimitivePtr &prim,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto query_type = input_args[kIndex0]->GetType();

  auto attention_out_type = query_type;
  auto softmax_lse_type = std::make_shared<TensorType>(kFloat32);
  bool has_deqScale1 = !input_args[kFusedInferAttentionScoreInputDequantScale1Index]->GetType()->isa<TypeNone>();
  bool has_qScale1 = !input_args[kFusedInferAttentionScoreInputQuantScale1Index]->GetType()->isa<TypeNone>();
  bool has_deqScale2 = !input_args[kFusedInferAttentionScoreInputDequantScale2Index]->GetType()->isa<TypeNone>();
  bool has_qScale2 = !input_args[kFusedInferAttentionScoreInputQuantScale2Index]->GetType()->isa<TypeNone>();
  bool has_qOffset2 = !input_args[kFusedInferAttentionScoreInputQuantOffset2Index]->GetType()->isa<TypeNone>();
  if (query_type->type_id() == TypeId::kNumberTypeInt8) {
    if (has_deqScale1 && has_qScale1 && has_deqScale2 && !has_qScale2 && !has_qOffset2) {
      attention_out_type = std::make_shared<TensorType>(kFloat16);
    }
  } else {
    attention_out_type = has_qScale2 ? std::make_shared<TensorType>(kInt8) : query_type;
  }
  return std::make_shared<Tuple>(std::vector<TypePtr>{attention_out_type, softmax_lse_type});
}
}  // namespace ops
}  // namespace mindspore
