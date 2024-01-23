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

#include "ops/flash_attention_score.h"

#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kFlashAttentionScoreSoftmaxLastDim = 8;
constexpr size_t kInputFlashAttentionScoreQueryBSHRank = 3;
constexpr size_t kInputFlashAttentionScoreQueryBNSDRank = 4;
constexpr char kInputFlashAttentionScoreLayoutBSH[] = "BSH";
constexpr char kInputFlashAttentionScoreLayoutBNSD[] = "BNSD";

// None indicates that the optional input is not passed
bool IsFlashAttentionScoreOptionalInputNotPass(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->BuildType()->type_id() == kMetaTypeNone;
}

abstract::TupleShapePtr FlashAttentionScoreInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kFlashAttentionScoreInputsNum, op_name);

  int64_t batch_size;
  int64_t seq_len;
  int64_t head_num = GetValue<int64_t>(primitive->GetAttr("head_num"));
  auto query_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kFlashAttentionScoreInputQueryIndex]->BuildShape())[kShape];
  auto input_layout = GetValue<std::string>(primitive->GetAttr("input_layout"));
  if (input_layout == kInputFlashAttentionScoreLayoutBSH) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBSHRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFlashAttentionScoreQueryBSHRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[0];
    seq_len = query_shape[1];
    auto hidden_size = query_shape[2];
    if (hidden_size % head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << H << " and " << N;
    }
  } else if (input_layout == kInputFlashAttentionScoreLayoutBNSD) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBNSDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFlashAttentionScoreQueryBNSDRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    if (head_num != query_shape[kIndex1]) {
      MS_LOG(EXCEPTION) << op_name << ": query_shape[1] must be equal to attribute 'head_num', but got "
                        << query_shape[1] << " and " << head_num;
    }
    seq_len = query_shape[kIndex2];
  } else {
    MS_LOG(EXCEPTION) << op_name << ": The value of attribute 'input_layout' must be one of ["
                      << kInputFlashAttentionScoreLayoutBNSD << ", " << kInputFlashAttentionScoreLayoutBSH
                      << "], but got " << input_layout;
  }
  abstract::BaseShapePtrList output_shape_ptr_list(kFlashAttentionScoreOutputsNum);
  output_shape_ptr_list[kFlashAttentionScoreOutputAttentionOutIndex] = std::make_shared<abstract::Shape>(query_shape);
  output_shape_ptr_list[kFlashAttentionScoreOutputSoftmaxMaxIndex] =
    std::make_shared<abstract::Shape>(ShapeVector{batch_size, head_num, seq_len, kFlashAttentionScoreSoftmaxLastDim});
  output_shape_ptr_list[kFlashAttentionScoreOutputSoftmaxSumIndex] =
    std::make_shared<abstract::Shape>(ShapeVector{batch_size, head_num, seq_len, kFlashAttentionScoreSoftmaxLastDim});
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TuplePtr FlashAttentionScoreInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kFloat16, kFloat32, kBFloat16};
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  // "x", "kernel_query", "kernel_key", "kernel_value", "gamma", " beta", "bias_query", "bias_key", "bias_value"
  (void)types.emplace("query", input_args[kFlashAttentionScoreInputQueryIndex]->BuildType());
  (void)types.emplace("key", input_args[kFlashAttentionScoreInputKeyIndex]->BuildType());
  (void)types.emplace("value", input_args[kFlashAttentionScoreInputValueIndex]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputPaddingMaskIndex])) {
    MS_LOG(EXCEPTION) << op_name << ": 'padding_mask' must be None currently.";
  }
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputRealShiftIndex])) {
    MS_LOG(EXCEPTION) << op_name << ": 'real_shift' must be None currently.";
  }
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputPrefix])) {
    MS_LOG(EXCEPTION) << op_name << ": 'prefix' must be None currently.";
  }

  auto keep_prob_value_ptr = prim->GetAttr("keep_prob");
  MS_EXCEPTION_IF_NULL(keep_prob_value_ptr);
  auto keep_prob = GetValue<float>(keep_prob_value_ptr);
  if (keep_prob > 1 || keep_prob < 0) {
    MS_LOG(EXCEPTION) << op_name << ": attribute `keep_prob` must be a floating point number in [0, 1], but got "
                      << keep_prob;
  }
  if (common::IsFloatEqual(keep_prob, 1.0)) {
    if (!IsFlashAttentionScoreOptionalInputNotPass(input_args[kFlashAttentionScoreInputDropMaskIndex])) {
      MS_LOG(EXCEPTION) << op_name << ": 'drop_mask' must be None when keep_prob is 1.0.";
    }
  } else {
    auto drop_mask_type = input_args[kFlashAttentionScoreInputDropMaskIndex]->BuildType();
    CheckAndConvertUtils::CheckTensorTypeValid("drop_mask", drop_mask_type, {kUInt8}, op_name);
  }

  TypePtrList output_type_ptr_list(kFlashAttentionScoreOutputsNum);
  output_type_ptr_list[kFlashAttentionScoreOutputAttentionOutIndex] = type;
  output_type_ptr_list[kFlashAttentionScoreOutputSoftmaxMaxIndex] = kFloat32;
  output_type_ptr_list[kFlashAttentionScoreOutputSoftmaxSumIndex] = kFloat32;
  return std::make_shared<Tuple>(output_type_ptr_list);
}
}  // namespace

AbstractBasePtr FlashAttentionScoreInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kFlashAttentionScoreInputsNum, primitive->name());
  auto infer_type = FlashAttentionScoreInferType(primitive, input_args);
  auto infer_shape = FlashAttentionScoreInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(FlashAttentionScore, BaseOperator);

// AG means auto generated
class MIND_API AGFlashAttentionScoreInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FlashAttentionScoreInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FlashAttentionScoreInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FlashAttentionScoreInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FlashAttentionScore, prim::kPrimFlashAttentionScore, AGFlashAttentionScoreInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
