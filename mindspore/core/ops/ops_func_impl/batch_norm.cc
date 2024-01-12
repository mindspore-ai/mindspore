/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/batch_norm.h"

#include <memory>
#include <utility>

#include "abstract/dshape.h"
#include "ops/op_def.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void MultiTimesClone(std::vector<T> *const vec, const T &ori, const size_t times) {
  for (size_t i = 0; i < times; ++i) {
    vec->push_back(ori->Clone());
  }
}

bool MeanAndVarianceValid(const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int> params_ids = {3, 4};
  size_t valid_param = 0;
  for (auto idx : params_ids) {
    auto type = input_args[IntToSize(idx)]->GetType();
    if (type->isa<TensorType>()) {
      auto tensor_type = type->cast<TensorTypePtr>();
      auto element = tensor_type->element();
      valid_param += element->type_id() != kMetaTypeNone ? 1 : 0;
    }
  }
  return valid_param == params_ids.size();
}

void BatchNormShapeCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                         const ShapeVector &x_shape, const ShapeVector &scale_shape, const ShapeVector &bias_shape,
                         const size_t attr_pos) {
  if (MS_LIKELY(!IsDynamicRank(x_shape))) {
    MS_CHECK_VALUE(2 <= x_shape.size() && x_shape.size() <= 4,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("rank of images", SizeToLong(x_shape.size()),
                                                               kIncludeBoth, {2, 4}, primitive));
  }
  MS_CHECK_VALUE(scale_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                            "rank of scale", SizeToLong(scale_shape.size()), kEqual, 1, primitive));
  MS_CHECK_VALUE(bias_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "rank of bias", SizeToLong(bias_shape.size()), kEqual, 1, primitive));
  if (MS_LIKELY(!(IsDynamic(scale_shape) || IsDynamic(bias_shape)))) {
    MS_CHECK_VALUE(bias_shape == scale_shape,
                   CheckAndConvertUtils::FormatCheckMsg("scale and bias", scale_shape, kEqual, bias_shape, primitive));
  }

  auto format_opt = GetScalarValue<int64_t>(input_args[attr_pos + 3]->GetValue());
  if (MS_LIKELY(format_opt.has_value() && !IsDynamic(x_shape) && !IsDynamic(scale_shape))) {
    mindspore::Format format = static_cast<mindspore::Format>(format_opt.value());
    auto channel = (format == Format::NCHW) ? x_shape[kInputIndex1] : x_shape.back();
    if (MS_UNLIKELY(scale_shape[kInputIndex0] != channel)) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << ", scale.shape[0] should be equal to input_x's channel dimension(" << channel
                               << "), bug got scale.shape[0]: " << scale_shape[kInputIndex0] << ".";
    }
  }

  if (!MeanAndVarianceValid(input_args)) {
    return;
  }
  auto mean_shape = input_args[kInputIndex3]->GetShape()->GetShapeVector();
  auto variance_shape = input_args[kInputIndex4]->GetShape()->GetShapeVector();
  MS_CHECK_VALUE(mean_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "rank of mean", SizeToLong(mean_shape.size()), kEqual, 1, primitive));
  MS_CHECK_VALUE(variance_shape.size() == 1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of variance", SizeToLong(variance_shape.size()),
                                                             kEqual, 1, primitive));
  auto is_training_opt = GetScalarValue<bool>(input_args[attr_pos + 0]->GetValue());
  if (MS_UNLIKELY(!is_training_opt.has_value())) {
    return;
  }
  auto is_training = is_training_opt.value();
  if (!is_training && !IsDynamic(mean_shape) && !IsDynamic(variance_shape) && !IsDynamic(scale_shape)) {
    if ((mean_shape[0] != variance_shape[0]) || (variance_shape[0] != scale_shape[0])) {
      MS_EXCEPTION(ValueError)
        << "For '" << primitive->name()
        << "', 'scale', 'bias', 'mean', and 'variance' should have the same size during training, but got "
        << scale_shape[0] << ", " << bias_shape[0] << ", " << mean_shape[0] << " and " << variance_shape[0] << ".";
    }
  }
}
}  // namespace
BaseShapePtr BatchNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto scale_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  auto bias_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  auto attr_pos = GetAttrPosZero();
  BatchNormShapeCheck(primitive, input_args, x_shape, scale_shape, bias_shape, attr_pos);

  auto x_shape_ptr = std::make_shared<abstract::TensorShape>(x_shape);
  auto scale_shape_ptr = IsDynamicRank(scale_shape)
                           ? std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeDimAny})
                           : std::make_shared<abstract::TensorShape>(scale_shape);
  std::vector<abstract::BaseShapePtr> shapes{std::move(x_shape_ptr)};
  MultiTimesClone<abstract::BaseShapePtr>(&shapes, scale_shape_ptr, 4);
  return std::make_shared<abstract::TupleShape>(std::move(shapes));
}

TypePtr BatchNormFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  const auto &x_type = input_args[0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim_name);

  std::map<std::string, TypePtr> check_types;
  auto scale_type = input_args[kInputIndex1]->GetType();
  (void)check_types.emplace("scale", input_args[kInputIndex1]->GetType());
  (void)check_types.emplace("bias", input_args[kInputIndex2]->GetType());
  if (MeanAndVarianceValid(input_args)) {
    (void)check_types.emplace("mean", input_args[kInputIndex3]->GetType());
    (void)check_types.emplace("variance", input_args[kInputIndex4]->GetType());
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(check_types, valid_types, prim_name);

  std::vector<TypePtr> out_types{x_type->Clone()};
  MultiTimesClone<TypePtr>(&out_types, scale_type, 4);
  return std::make_shared<Tuple>(std::move(out_types));
}

int32_t BatchNormFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const size_t attr_pos = GetAttrPosZero();
  auto epsilon_value = GetScalarValue<pyfloat>(input_args[attr_pos + 1]->GetValue());
  if (MS_UNLIKELY(!epsilon_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(epsilon_value.value() > 0 && epsilon_value.value() <= 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<pyfloat>("epsilon", epsilon_value.value(), kIncludeRight,
                                                                      {0., 1.}, primitive));

  auto momentum_value = GetScalarValue<pyfloat>(input_args[attr_pos + 2]->GetValue());
  if (MS_UNLIKELY(!momentum_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto momentum = momentum_value.value();
  MS_CHECK_VALUE(momentum >= 0 && momentum <= 1, CheckAndConvertUtils::FormatCheckInRangeMsg<pyfloat>(
                                                   "momentum", momentum, kIncludeRight, {0., 1.}, primitive));

  auto format_opt = GetScalarValue<int64_t>(input_args[attr_pos + 3]->GetValue());
  if (MS_UNLIKELY(!format_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  mindspore::Format format = static_cast<mindspore::Format>(format_opt.value());
  if (MS_UNLIKELY(format != Format::NCHW && format != Format::NHWC)) {
    MS_LOG(EXCEPTION) << "The data format value " << FormatEnumToString(format) << " is invalid, " << primitive->name()
                      << " only support NCHW and NHWC.";
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
