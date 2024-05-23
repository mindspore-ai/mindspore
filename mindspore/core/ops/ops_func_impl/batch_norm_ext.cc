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

#include "ops/ops_func_impl/batch_norm_ext.h"

#include <memory>
#include <utility>

#include "abstract/dshape.h"
#include "ops/op_def.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void MultiClone(std::vector<T> *const vec, const T &ori, const size_t times) {
  for (size_t i = 0; i < times; ++i) {
    vec->push_back(ori->Clone());
  }
}

void BatchNormExtShapeCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                            const ShapeVector &x_shape, const ShapeVector &weight_shape, const ShapeVector &bias_shape,
                            const size_t attr_pos) {
  if (MS_LIKELY(!IsDynamicRank(x_shape))) {
    MS_CHECK_VALUE(2 <= x_shape.size() && x_shape.size() <= 8,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("rank of images", SizeToLong(x_shape.size()),
                                                               kIncludeBoth, {2, 8}, primitive));
  }
  MS_CHECK_VALUE(weight_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                             "rank of weight", SizeToLong(weight_shape.size()), kEqual, 1, primitive));
  MS_CHECK_VALUE(bias_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "rank of bias", SizeToLong(bias_shape.size()), kEqual, 1, primitive));
  if (MS_LIKELY(!(IsDynamic(weight_shape) || IsDynamic(bias_shape)))) {
    MS_CHECK_VALUE(bias_shape == weight_shape, CheckAndConvertUtils::FormatCheckMsg("weight and bias", weight_shape,
                                                                                    kEqual, bias_shape, primitive));
  }
  if (MS_LIKELY(!IsDynamic(x_shape) && !IsDynamic(weight_shape))) {
    auto channel = x_shape[kInputIndex1];
    if (MS_UNLIKELY(weight_shape[kInputIndex0] != channel)) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << ", weight.shape[0] should be equal to input_x's channel dimension: " << channel
                               << ", bug got weight.shape[0]: " << weight_shape[kInputIndex0] << ".";
    }
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
  if (!is_training && !IsDynamic(mean_shape) && !IsDynamic(variance_shape) && !IsDynamic(weight_shape)) {
    if ((mean_shape[0] != variance_shape[0]) || (variance_shape[0] != weight_shape[0])) {
      MS_EXCEPTION(ValueError)
        << "For '" << primitive->name()
        << "', 'weight', 'bias', 'running_mean', and 'running_var' should have the same size during training, but got "
        << weight_shape[0] << ", " << bias_shape[0] << ", " << mean_shape[0] << " and " << variance_shape[0] << ".";
    }
  }
}
}  // namespace
BaseShapePtr BatchNormExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  const auto &weight_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  const auto &bias_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  auto attr_pos = GetAttrPosZero();
  BatchNormExtShapeCheck(primitive, input_args, x_shape, weight_shape, bias_shape, attr_pos);

  auto x_shape_ptr = std::make_shared<abstract::TensorShape>(x_shape);
  auto weight_shape_ptr = IsDynamicRank(weight_shape)
                            ? std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeDimAny})
                            : std::make_shared<abstract::TensorShape>(weight_shape);
  std::vector<abstract::BaseShapePtr> shapes{std::move(x_shape_ptr)};
  MultiClone<abstract::BaseShapePtr>(&shapes, weight_shape_ptr, 2);
  return std::make_shared<abstract::TupleShape>(std::move(shapes));
}

TypePtr BatchNormExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_type = input_args[kInputIndex0]->GetType();
  auto weight_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(weight_type);
  std::vector<TypePtr> types_list;
  types_list = {x_type, weight_type, weight_type};
  return std::make_shared<Tuple>(types_list);
}

ShapeArray BatchNormExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &weight_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(weight_tensor);
  return {x_tensor->shape(), weight_tensor->shape(), weight_tensor->shape()};
}

TypePtrList BatchNormExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &weight_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(weight_tensor);
  return {x_tensor->Dtype(), weight_tensor->Dtype(), weight_tensor->Dtype()};
}

REGISTER_SIMPLE_INFER(kNameBatchNormExt, BatchNormExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
