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

#include <set>
#include <string>
#include <sstream>
#include "ops/prompt_flash_attention.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputQueryBSHRank = 3;
constexpr size_t kInputQueryBNSDRank = 4;

ShapeValueDType GetDimension(const std::vector<ShapeValueDType> &dimensions, const std::string &op_name,
                             const std::string &input_name) {
  if (dimensions.empty()) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the " << input_name << " should not be empty";
    return abstract::Shape::kShapeDimAny;
  }
  ShapeValueDType baseValue = abstract::Shape::kShapeDimAny;
  for (const auto &item : dimensions) {
    if (item == abstract::Shape::kShapeDimAny || item == baseValue) {
      continue;
    }
    if (baseValue == abstract::Shape::kShapeDimAny && item > 0) {
      baseValue = item;
    } else {
      std::ostringstream buffer;
      for (const auto &dim : dimensions) {
        buffer << dim << ", ";
      }
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the " << input_name << " should not be equal -1 or equal"
                        << baseValue << " but got " << buffer.str();
    }
  }
  return baseValue;
}

// None indicates that the optional input is not passed
bool IsOptionalInputNotPass(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->BuildType()->type_id() == kMetaTypeNone;
}

abstract::TupleShapePtr PromptFlashAttentionInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPromptFlashAttentionInputsNum, op_name);

  auto input_layout = GetValue<std::string>(primitive->GetAttr("input_layout"));
  auto num_heads = GetValue<int64_t>(primitive->GetAttr("num_heads"));
  if (num_heads == 0) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the num_heads should not be zero.";
  }
  auto query_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kPromptFlashAttentionInputQueryIndex]->BuildShape())[kShape];
  auto key_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kPromptFlashAttentionInputKeyIndex]->BuildShape())[kShape];
  auto value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kPromptFlashAttentionInputValueIndex]->BuildShape())[kShape];
  auto atten_mask_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kPromptFlashAttentionInputAttnMaskIndex]->BuildShape())[kShape];

  bool qeury_rank_is_dyn = IsDynamicRank(query_shape);
  bool key_rank_is_dyn = IsDynamicRank(key_shape);
  bool value_rank_is_dyn = IsDynamicRank(value_shape);
  bool atten_mask_rank_is_dyn = IsDynamicRank(atten_mask_shape);
  size_t temp_rank = input_layout == "BSH" ? kInputQueryBSHRank : kInputQueryBNSDRank;
  if (qeury_rank_is_dyn) {
    query_shape = std::vector(temp_rank, abstract::Shape::kShapeDimAny);
  }
  if (key_rank_is_dyn) {
    key_shape = std::vector(temp_rank, abstract::Shape::kShapeDimAny);
  }
  if (value_rank_is_dyn) {
    value_shape = std::vector(temp_rank, abstract::Shape::kShapeDimAny);
  }
  if (atten_mask_rank_is_dyn) {
    atten_mask_shape = std::vector(temp_rank, abstract::Shape::kShapeDimAny);
  }
  CheckAndConvertUtils::CheckInteger("rank of query", query_shape.size(), kEqual, temp_rank, op_name);
  CheckAndConvertUtils::CheckInteger("rank of key", key_shape.size(), kEqual, temp_rank, op_name);
  CheckAndConvertUtils::CheckInteger("rank of value", value_shape.size(), kEqual, temp_rank, op_name);
  auto atten_mask_shape_size = atten_mask_shape.size();
  if (atten_mask_shape_size != kInputQueryBSHRank && atten_mask_shape_size != kInputQueryBNSDRank) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the rank of atten_mask should be 3 or 4, but got "
                      << SizeToLong(atten_mask_shape_size);
  }
  ShapeVector attention_out_shape(temp_rank, abstract::Shape::kShapeDimAny);
  if (input_layout == "BSH") {
    auto b_index = 0;
    auto s_index = 1;
    auto h_index = 2;

    attention_out_shape[b_index] =
      GetDimension({query_shape[b_index], key_shape[b_index], value_shape[b_index]}, op_name, "B");
    attention_out_shape[s_index] =
      GetDimension({query_shape[s_index], atten_mask_shape[atten_mask_shape_size - 2]}, op_name, "Q_S");
    (void)GetDimension({key_shape[s_index], value_shape[s_index], atten_mask_shape[atten_mask_shape_size - 1]}, op_name,
                       "KV_S");
    attention_out_shape[h_index] =
      GetDimension({query_shape[h_index], key_shape[h_index], value_shape[h_index]}, op_name, "H");
    auto h_dim = attention_out_shape[h_index];
    if (h_dim % num_heads != 0) {
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], H must be divisible by `num_heads`, but got " << h_dim
                        << " and " << num_heads;
    }
  } else if (input_layout == "BNSD") {
    auto b_index = 0;
    auto n_index = 1;
    auto s_index = 2;
    auto d_index = 3;

    attention_out_shape[b_index] =
      GetDimension({query_shape[b_index], key_shape[b_index], value_shape[b_index]}, op_name, "B");
    attention_out_shape[n_index] =
      GetDimension({query_shape[n_index], key_shape[n_index], value_shape[n_index]}, op_name, "N");
    attention_out_shape[s_index] =
      GetDimension({query_shape[s_index], atten_mask_shape[atten_mask_shape_size - 2]}, op_name, "Q_S");
    (void)GetDimension({key_shape[s_index], value_shape[s_index], atten_mask_shape[atten_mask_shape_size - 1]}, op_name,
                       "KV_S");
    attention_out_shape[d_index] =
      GetDimension({query_shape[d_index], key_shape[d_index], value_shape[d_index]}, op_name, "D");
    auto n_dim = attention_out_shape[n_index];
    if (n_dim != num_heads) {
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], N must equal num_heads, but got " << n_dim << " and "
                        << num_heads;
    }
  } else {
    attention_out_shape.resize(1);
    attention_out_shape[0] = abstract::Shape::kShapeRankAny;
  }
  abstract::BaseShapePtrList output_shape_ptr_list(kPromptFlashAttentionOutputsNum);
  output_shape_ptr_list[kPromptFlashAttentionOutputAttentionOutIndex] =
    std::make_shared<abstract::Shape>(attention_out_shape);
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TuplePtr PromptFlashAttentionInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("query", input_args[kPromptFlashAttentionInputQueryIndex]->BuildType());
  (void)types.emplace("key", input_args[kPromptFlashAttentionInputKeyIndex]->BuildType());
  (void)types.emplace("value", input_args[kPromptFlashAttentionInputValueIndex]->BuildType());
  (void)types.emplace("attn_mask", input_args[kPromptFlashAttentionInputAttnMaskIndex]->BuildType());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  if (!IsOptionalInputNotPass(input_args[kPromptFlashAttentionInputPaddingMaskIndex])) {
    MS_LOG(EXCEPTION) << "For " << op_name << ": 'padding_mask' must be None currently.";
  }
  TypePtrList output_type_ptr_list(kPromptFlashAttentionOutputsNum);
  output_type_ptr_list[kPromptFlashAttentionOutputAttentionOutIndex] = type;
  return std::make_shared<Tuple>(output_type_ptr_list);
}
}  // namespace

AbstractBasePtr PromptFlashAttentionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPromptFlashAttentionInputsNum, primitive->name());
  auto infer_shape = PromptFlashAttentionInferShape(primitive, input_args);
  auto infer_type = PromptFlashAttentionInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(PromptFlashAttention, BaseOperator);

// AG means auto generated
class MIND_API AGPromptFlashAttentionInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PromptFlashAttentionInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PromptFlashAttentionInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PromptFlashAttentionInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PromptFlashAttention, prim::kPrimPromptFlashAttention, AGPromptFlashAttentionInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
