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

#include "ops/grad/flash_attention_score_grad.h"

#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
enum FlashAttentionScoreInputIndex : size_t {
  kFlashAttentionScoreGradInputQueryIndex = 0,
  kFlashAttentionScoreGradInputKeyIndex,
  kFlashAttentionScoreGradInputValueIndex,
  kFlashAttentionScoreGradInputAttnMaskIndex,
  kFlashAttentionScoreGradInputAttentionInIndex,
  kFlashAttentionScoreGradInputSoftmaxMaxIndex,
  kFlashAttentionScoreGradInputSoftmaxSumIndex,
  kFlashAttentionScoreGradInputDyIndex,
  kFlashAttentionScoreGradInputDropMaskIndex,
  kFlashAttentionScoreGradInputRealShiftIndex,
  kFlashAttentionScoreGradInputPaddingMaskIndex,
  kFlashAttentionScoreGradInputSoftmaxOutIndex,
  kFlashAttentionScoreGradInputsNum,
};
enum FlashAttentionScoreOutputIndex : size_t {
  kFlashAttentionScoreGradOutputDqIndex = 0,
  kFlashAttentionScoreGradOutputDkIndex,
  kFlashAttentionScoreGradOutputDvIndex,
  kFlashAttentionScoreGradOutputsNum,
};
constexpr size_t kInputQueryDim = 3;
constexpr size_t kSoftmaxLastDim = 8;

// None indicates that the optional input is not passed
bool IsOptionalInputNotPass(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->BuildType()->type_id() == kMetaTypeNone;
}

void CheckInputShape(const AbstractBasePtr &input, const std::vector<ShapeValueDType> &expect_shape,
                     const std::string &op_name, const std::string &input_name, bool optional = false) {
  MS_EXCEPTION_IF_NULL(input);
  if (IsOptionalInputNotPass(input) && optional) {
    return;
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input->BuildShape())[kShape];
  if (input_shape != expect_shape) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be " << expect_shape
                      << ", but got shape is " << input_shape;
  }
}

abstract::TupleShapePtr FlashAttentionScoreGradInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kFlashAttentionScoreGradInputsNum, op_name);
  auto query_shape_map =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kFlashAttentionScoreGradInputQueryIndex]->BuildShape());
  auto query_shape = query_shape_map[kShape];
  if (query_shape.size() != kInputQueryDim) {
    MS_LOG(EXCEPTION) << op_name << ": The rank of input `query` must be " << kInputQueryDim << ", but got "
                      << query_shape.size();
  }
  auto B = query_shape[kIndex0];
  auto S = query_shape[kIndex1];
  auto H = query_shape[kIndex2];
  auto N = GetValue<int64_t>(primitive->GetAttr("head_num"));
  if (H % N != 0) {
    MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << H << " and " << N;
  }
  CheckInputShape(input_args[kFlashAttentionScoreGradInputKeyIndex], {B, S, H}, op_name, "key");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputValueIndex], {B, S, H}, op_name, "value");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputAttnMaskIndex], {B, 1, S, S}, op_name, "attn_mask");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputDropMaskIndex], {B, N, S, S / 8}, op_name, "drop_mask", true);
  CheckInputShape(input_args[kFlashAttentionScoreGradInputSoftmaxMaxIndex], {B, N, S, kSoftmaxLastDim}, op_name,
                  "softmax_max");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputSoftmaxSumIndex], {B, N, S, kSoftmaxLastDim}, op_name,
                  "softmax_sum");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputSoftmaxOutIndex], {B, N, S, S}, op_name, "softmax_out", true);
  CheckInputShape(input_args[kFlashAttentionScoreGradInputAttentionInIndex], {B, S, H}, op_name, "attention_in");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputDyIndex], {B, S, H}, op_name, "dy");

  abstract::BaseShapePtrList output_shape_ptr_list(kFlashAttentionScoreGradOutputsNum);
  output_shape_ptr_list[kFlashAttentionScoreGradOutputDqIndex] =
    std::make_shared<abstract::Shape>(std::vector<ShapeValueDType>{B, S, H});
  output_shape_ptr_list[kFlashAttentionScoreGradOutputDkIndex] =
    std::make_shared<abstract::Shape>(std::vector<ShapeValueDType>{B, S, H});
  output_shape_ptr_list[kFlashAttentionScoreGradOutputDvIndex] =
    std::make_shared<abstract::Shape>(std::vector<ShapeValueDType>{B, S, H});
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TuplePtr FlashAttentionScoreGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types1;
  std::map<std::string, TypePtr> types2;
  // "x", "kernel_query", "kernel_key", "kernel_value", "gamma", " beta", "bias_query", "bias_key", "bias_value"
  (void)types1.emplace("query", input_args[kFlashAttentionScoreGradInputQueryIndex]->BuildType());
  (void)types1.emplace("key", input_args[kFlashAttentionScoreGradInputKeyIndex]->BuildType());
  (void)types1.emplace("value", input_args[kFlashAttentionScoreGradInputValueIndex]->BuildType());
  auto attn_mask_type = input_args[kFlashAttentionScoreGradInputAttnMaskIndex]->BuildType();
  CheckAndConvertUtils::CheckTensorTypeValid("attn_mask", attn_mask_type, {kFloat16, kUInt8}, op_name);
  if (!IsOptionalInputNotPass(input_args[kFlashAttentionScoreGradInputPaddingMaskIndex])) {
    MS_LOG(EXCEPTION) << op_name << ": 'padding_mask' must be None currently.";
  }
  if (!IsOptionalInputNotPass(input_args[kFlashAttentionScoreGradInputRealShiftIndex])) {
    MS_LOG(EXCEPTION) << op_name << ": 'real_shift' must be None currently.";
  }
  (void)types1.emplace("attention_in", input_args[kFlashAttentionScoreGradInputAttentionInIndex]->BuildType());
  (void)types2.emplace("softmax_max", input_args[kFlashAttentionScoreGradInputSoftmaxMaxIndex]->BuildType());
  (void)types2.emplace("softmax_sum", input_args[kFlashAttentionScoreGradInputSoftmaxSumIndex]->BuildType());
  if (!IsOptionalInputNotPass(input_args[kFlashAttentionScoreGradInputSoftmaxOutIndex])) {
    MS_LOG(EXCEPTION) << op_name << ": 'softmax_out' must be None currently.";
  }
  (void)types1.emplace("dy", input_args[kFlashAttentionScoreGradInputDyIndex]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types1, {kFloat16, kFloat32, kBFloat16}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types2, {kFloat32}, op_name);

  auto keep_prob_value_ptr = prim->GetAttr("keep_prob");
  MS_EXCEPTION_IF_NULL(keep_prob_value_ptr);
  auto keep_prob = GetValue<float>(keep_prob_value_ptr);
  if (keep_prob > 1 || keep_prob < 0) {
    MS_LOG(EXCEPTION) << op_name << ": attribute `keep_prob` must be a floating point number in [0, 1], but got "
                      << keep_prob;
  }
  if (common::IsFloatEqual(keep_prob, 1.0)) {
    if (!IsOptionalInputNotPass(input_args[kFlashAttentionScoreGradInputDropMaskIndex])) {
      MS_LOG(EXCEPTION) << op_name << ": 'drop_mask' must be None when keep_prob is 1.0.";
    }
  } else {
    auto drop_mask_type = input_args[kFlashAttentionScoreGradInputDropMaskIndex]->BuildType();
    CheckAndConvertUtils::CheckTensorTypeValid("drop_mask", drop_mask_type, {kUInt8}, op_name);
  }

  return std::make_shared<Tuple>(std::vector<TypePtr>(kFlashAttentionScoreGradOutputsNum, type));
}
}  // namespace

AbstractBasePtr FlashAttentionScoreGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kFlashAttentionScoreGradInputsNum, primitive->name());
  auto infer_type = FlashAttentionScoreGradInferType(primitive, input_args);
  auto infer_shape = FlashAttentionScoreGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(FlashAttentionScoreGrad, BaseOperator);

// AG means auto generated
class MIND_API AGFlashAttentionScoreGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FlashAttentionScoreGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FlashAttentionScoreGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FlashAttentionScoreGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FlashAttentionScoreGrad, prim::kPrimFlashAttentionScoreGrad,
                                 AGFlashAttentionScoreGradInfer, false);
}  // namespace ops
}  // namespace mindspore
