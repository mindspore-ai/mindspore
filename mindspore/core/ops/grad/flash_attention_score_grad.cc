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
  kFlashAttentionScoreGradInputPrefixIndex,
  kFlashAttentionScoreGradInputsNum,
};
enum FlashAttentionScoreOutputIndex : size_t {
  kFlashAttentionScoreGradOutputDqIndex = 0,
  kFlashAttentionScoreGradOutputDkIndex,
  kFlashAttentionScoreGradOutputDvIndex,
  kFlashAttentionScoreGradOutputsNum,
};
constexpr size_t kSoftmaxLastDim = 8;
constexpr size_t kInputQueryBSHRank = 3;
constexpr size_t kInputQueryBNSDRank = 4;
constexpr char kInputLayoutBSH[] = "BSH";
constexpr char kInputLayoutBNSD[] = "BNSD";

// None indicates that the optional input is not passed
bool IsOptionalInputNotPass(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->BuildType()->type_id() == kMetaTypeNone;
}

void CheckInputShape(const AbstractBasePtr &input, const ShapeVector &expect_shape, const std::string &op_name,
                     const std::string &input_name, bool optional = false) {
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

void CheckInputShape(const AbstractBasePtr &input, const std::vector<ShapeVector> &expect_shape_list,
                     const std::string &op_name, const std::string &input_name, bool optional = false) {
  MS_EXCEPTION_IF_NULL(input);
  if (IsOptionalInputNotPass(input) && optional) {
    return;
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input->BuildShape())[kShape];
  if (!std::any_of(expect_shape_list.begin(), expect_shape_list.end(),
                   [&input_shape](const ShapeVector &expect_shape) { return input_shape != expect_shape; })) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be one of " << expect_shape_list
                      << ", but got shape is " << input_shape;
  }
}

abstract::TupleShapePtr FlashAttentionScoreGradInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kFlashAttentionScoreGradInputsNum, op_name);

  int64_t batch_size = 0;
  int64_t seq_len = 0;
  int64_t head_size = 0;
  int64_t head_num = GetValue<int64_t>(primitive->GetAttr("head_num"));
  auto input_layout = GetValue<std::string>(primitive->GetAttr("input_layout"));
  auto query_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kFlashAttentionScoreGradInputQueryIndex]->BuildShape())[kShape];
  ShapeVector expect_q_shape;
  std::vector<ShapeVector> expect_kv_shape;
  if (input_layout == kInputLayoutBSH) {
    if (query_shape.size() != kInputQueryBSHRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of input `query` must be " << kInputQueryBSHRank << ", but got "
                        << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    seq_len = query_shape[kIndex1];
    auto hidden_size = query_shape[kIndex2];
    head_num = GetValue<int64_t>(primitive->GetAttr("head_num"));
    if (hidden_size % head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << hidden_size
                        << " and " << head_num;
    }
    expect_q_shape = {batch_size, seq_len, hidden_size};
    expect_kv_shape = {{batch_size, seq_len, hidden_size}, {batch_size, seq_len, head_size}};
  } else if (input_layout == kInputLayoutBNSD) {
    if (query_shape.size() != kInputQueryBNSDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputQueryBNSDRank << ", but got "
                        << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    if (head_num != query_shape[kIndex1]) {
      MS_LOG(EXCEPTION) << op_name << ": query_shape[1] must be equal to attribute 'head_num', but got "
                        << query_shape[1] << " and " << head_num;
    }
    seq_len = query_shape[kIndex2];
    head_size = query_shape[kIndex3];
    expect_q_shape = {batch_size, head_num, seq_len, head_size};
    expect_kv_shape = {{batch_size, head_num, seq_len, head_size}, {batch_size, 1, seq_len, head_size}};
  } else {
    MS_LOG(EXCEPTION) << op_name << ": The value of attribute 'input_layout' must be one of [" << kInputLayoutBNSD
                      << ", " << kInputLayoutBSH << "], but got " << input_layout;
  }
  CheckInputShape(input_args[kFlashAttentionScoreGradInputKeyIndex], expect_kv_shape, op_name, "key");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputValueIndex], expect_kv_shape, op_name, "value");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputAttentionInIndex], expect_q_shape, op_name, "attention_in");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputDyIndex], expect_q_shape, op_name, "dy");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputAttnMaskIndex], {batch_size, 1, seq_len, seq_len}, op_name,
                  "attn_mask");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputDropMaskIndex], {batch_size, head_num, seq_len, seq_len / 8},
                  op_name, "drop_mask", true);
  CheckInputShape(input_args[kFlashAttentionScoreGradInputSoftmaxMaxIndex],
                  {batch_size, head_num, seq_len, kSoftmaxLastDim}, op_name, "softmax_max");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputSoftmaxSumIndex],
                  {batch_size, head_num, seq_len, kSoftmaxLastDim}, op_name, "softmax_sum");
  CheckInputShape(input_args[kFlashAttentionScoreGradInputSoftmaxOutIndex], {batch_size, head_num, seq_len, seq_len},
                  op_name, "softmax_out", true);

  abstract::BaseShapePtrList output_shape_ptr_list(kFlashAttentionScoreGradOutputsNum);
  output_shape_ptr_list[kFlashAttentionScoreGradOutputDqIndex] = std::make_shared<abstract::Shape>(query_shape);
  auto key_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kFlashAttentionScoreGradInputKeyIndex]->BuildShape())[kShape];
  output_shape_ptr_list[kFlashAttentionScoreGradOutputDkIndex] = std::make_shared<abstract::Shape>(key_shape);
  auto value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kFlashAttentionScoreGradInputValueIndex]->BuildShape())[kShape];
  output_shape_ptr_list[kFlashAttentionScoreGradOutputDvIndex] = std::make_shared<abstract::Shape>(value_shape);
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
  CheckAndConvertUtils::CheckTensorTypeValid("attn_mask", attn_mask_type, {kUInt8}, op_name);
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
  if (!IsOptionalInputNotPass(input_args[kFlashAttentionScoreGradInputPrefixIndex])) {
    MS_LOG(EXCEPTION) << op_name << ": 'prefix' must be None currently.";
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
