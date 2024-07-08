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
#include "ops/ops_func_impl/group_norm.h"
#include <map>
#include <string>
#include <set>
#include <memory>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
constexpr int64_t kNumberTwo = 2;
constexpr int64_t kNumberEight = 8;
BaseShapePtr GroupNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  const auto &x_shape = x_shape_ptr->GetShapeVector();
  const auto &weight_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  const auto &bias_shape = input_args[kInputIndex3]->GetShape()->GetShapeVector();
  auto num_groups_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  std::vector<BaseShapePtr> shapes_list;
  if (!num_groups_opt.has_value() || IsDynamicRank(x_shape)) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  int64_t num_groups = num_groups_opt.value();
  const auto x_rank = x_shape.size();
  if (x_rank < kNumberTwo || x_rank > kNumberEight) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', The dim of input must be between 2 and 8. But got: " << x_rank << ".";
  }
  if (weight_shape.size() == 0 || bias_shape.size() == 0) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name()
                            << ", the weight and bias must be a tensor, but got a number.";
  }
  MS_CHECK_VALUE(weight_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                             "rank of weight", SizeToLong(weight_shape.size()), kEqual, 1, primitive));
  MS_CHECK_VALUE(bias_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "rank of bias", SizeToLong(bias_shape.size()), kEqual, 1, primitive));
  if (MS_LIKELY(!(IsDynamic(weight_shape) || IsDynamic(bias_shape)))) {
    MS_CHECK_VALUE(bias_shape == weight_shape, CheckAndConvertUtils::FormatCheckMsg("weight and bias", weight_shape,
                                                                                    kEqual, bias_shape, primitive));
  }
  const int64_t N = x_shape[0];
  const int64_t channel = x_shape[1];
  if (!IsDynamic(x_shape) && (channel % num_groups != 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", the 'num_channels' must be divided by 'num_groups', "
                             << "but got 'num_channels': " << channel << " ,'num_groups': " << num_groups;
  }
  if (!IsDynamic(x_shape) && !IsDynamic(weight_shape) && MS_UNLIKELY(weight_shape[kInputIndex0] != channel)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", shape of weight and bias should be equal to input_x's channel dimension: " << channel
                             << ", bug got shape: " << weight_shape << ".";
  }
  ShapeVector out_shape{N, num_groups};
  (void)shapes_list.emplace_back(x_shape_ptr->Clone());
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(out_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(out_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr GroupNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  std::map<std::string, TypePtr> types;
  const auto &x_type = input_args[kInputIndex0]->GetType();
  const auto &weight_type = input_args[kInputIndex2]->GetType();
  const auto &bias_type = input_args[kInputIndex3]->GetType();
  (void)types.emplace("input", x_type);
  (void)types.emplace("weight", weight_type);
  (void)types.emplace("bias", bias_type);

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);

  std::vector<TypePtr> types_list;
  types_list = {x_type, x_type, x_type};
  return std::make_shared<Tuple>(types_list);
}

TypePtrList GroupNormFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &prim_name = primitive->name();
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &weight_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
  const auto &bias_tensor = input_values[kInputIndex3]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(weight_tensor);
  MS_EXCEPTION_IF_NULL(bias_tensor);

  const auto &x_type = x_tensor->Dtype();
  const auto &weight_type = weight_tensor->Dtype();
  const auto &bias_type = bias_tensor->Dtype();
  TypePtrList input_types{x_type, weight_type, bias_type};
  const std::set<TypePtr> types_cnt(input_types.begin(), input_types.end());
  auto is_valid = std::all_of(input_types.begin(), input_types.end(), [](const TypePtr &type) {
    return (type == kFloat16 || type == kFloat32 || type == kBFloat16);
  });
  if (!is_valid || types_cnt.size() != 1) {
    MS_EXCEPTION(TypeError) << "For " << prim_name
                            << ". input arguments' types must be the same and be one of [BFloat16, Float16, Float32]. "
                            << "But got input's type: " << x_type << ", weight's type: " << weight_type
                            << ", bias's type: " << bias_type << ".";
  }
  return {x_type, x_type, x_type};
}

ShapeArray GroupNormFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &weight_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
  const auto &bias_tensor = input_values[kInputIndex3]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(weight_tensor);
  MS_EXCEPTION_IF_NULL(bias_tensor);
  const auto &x_shape = x_tensor->shape();
  const auto &weight_shape = weight_tensor->shape();
  const auto &bias_shape = bias_tensor->shape();
  const auto &num_groups_value = input_values[kInputIndex1];
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    return {x_shape, x_shape, x_shape};
  }
  auto num_groups_opt = GetScalarValue<int64_t>(num_groups_value);
  if (MS_UNLIKELY(!num_groups_opt.has_value())) {
    ShapeVector dynamic_rank_shape{abstract::TensorShape::kShapeRankAny};
    return {dynamic_rank_shape, dynamic_rank_shape, dynamic_rank_shape};
  }
  const auto x_rank = x_shape.size();
  if (x_rank < kNumberTwo || x_rank > kNumberEight) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', The dim of input must be between 2 and 8. But got: " << x_rank << ".";
  }
  if (weight_shape.size() == 0 || bias_shape.size() == 0) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name()
                            << ", the weight and bias must be a tensor, but got a number.";
  }
  if (weight_shape.size() != 1 || bias_shape.size() != 1 || weight_shape != bias_shape) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ". "
                             << "Weight and bias must have the same shape and the rank of weight and bias must be 1. "
                             << "But got weight's shape: " << weight_shape << ", bias's shape: " << bias_shape;
  }
  int64_t num_groups = num_groups_opt.value();
  auto N = x_shape[0];
  const int64_t channel = x_shape[1];
  if (!IsDynamic(x_shape) && (channel % num_groups != 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", the 'num_channels' must be divided by 'num_groups', "
                             << "but got 'num_channels': " << channel << " ,'num_groups': " << num_groups;
  }
  if (!IsDynamic(x_shape) && !IsDynamic(weight_shape) && MS_UNLIKELY(weight_shape[kInputIndex0] != channel)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", shape of weight and bias should be equal to input_x's channel dimension: " << channel
                             << ", bug got shape: " << weight_shape << ".";
  }
  ShapeVector out_shape{N, num_groups_opt.value()};
  return {x_shape, out_shape, out_shape};
}

REGISTER_SIMPLE_INFER(kNameGroupNorm, GroupNormFuncImpl)
}  // namespace ops
}  // namespace mindspore
