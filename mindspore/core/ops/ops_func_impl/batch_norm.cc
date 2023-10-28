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

#include "ops/ops_func_impl/batch_norm.h"

#include <memory>
#include <utility>

#include "abstract/dshape.h"
#include "ops/manually_defined_ops_name.h"
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
    auto bak = ori->Clone();
    vec->push_back(std::move(bak));
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
    MS_CHECK_VALUE(
      scale_shape[kInputIndex0] == channel,
      CheckAndConvertUtils::FormatCheckIntegerMsg("channel", scale_shape[kInputIndex0], kEqual, channel, primitive));
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
  auto x_type = input_args[kInputIndex0]->GetType()->Clone();
  auto scale_type = input_args[kInputIndex1]->GetType();
  std::vector<TypePtr> types{std::move(x_type)};
  MultiTimesClone<TypePtr>(&types, scale_type, 4);
  return std::make_shared<Tuple>(std::move(types));
}

int32_t BatchNormFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const size_t attr_pos = GetAttrPosZero();
  auto epsilon_value = GetScalarValue<pyfloat>(input_args[attr_pos + 1]->GetValue());
  if (MS_UNLIKELY(!epsilon_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(epsilon_value.value() > 0 && epsilon_value.value() <= 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<double>("epsilon", epsilon_value.value(), kIncludeRight,
                                                                     {0., 1.}, primitive));

  auto momentum_value = GetScalarValue<pyfloat>(input_args[attr_pos + 2]->GetValue());
  if (MS_UNLIKELY(!momentum_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto momentum = momentum_value.value();
  MS_CHECK_VALUE(momentum >= 0 && momentum <= 1, CheckAndConvertUtils::FormatCheckInRangeMsg<double>(
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

auto gBatchNormFuncImpl = BatchNormFuncImpl();
OpDef gBatchNorm = {
  .name_ = kNameBatchNorm,
  .args_ = {{.arg_name_ = "input_x", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
            {.arg_name_ = "scale", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
            {.arg_name_ = "bias", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
            {.arg_name_ = "mean", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
            {.arg_name_ = "variance", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
            {.arg_name_ = "is_training", .arg_dtype_ = DT_BOOL, .as_init_arg_ = 1},
            {.arg_name_ = "epsilon", .arg_dtype_ = DT_FLOAT, .as_init_arg_ = 1},
            {.arg_name_ = "momentum", .arg_dtype_ = DT_FLOAT, .as_init_arg_ = 1},
            {.arg_name_ = "data_format", .arg_dtype_ = DT_INT, .as_init_arg_ = 1}},
  .returns_ = {{.arg_name_ = "output_x", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
               {.arg_name_ = "batch_mean", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
               {.arg_name_ = "batch_variance", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
               {.arg_name_ = "reserve_space_1", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
               {.arg_name_ = "reserve_space_2", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0}},
  .indexes_ = {{"input_x", 0},
               {"scale", 1},
               {"bias", 2},
               {"mean", 3},
               {"variance", 4},
               {"is_training", 5},
               {"epsilon", 6},
               {"momentum", 7},
               {"data_format", 8}},
  .func_impl_ = gBatchNormFuncImpl,
};

REGISTER_PRIMITIVE_OP_DEF(kNameBatchNorm, &gBatchNorm);
}  // namespace ops
}  // namespace mindspore
